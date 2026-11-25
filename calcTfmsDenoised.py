"""
This file contains the core part of this project's code. Using the denoised CT data, a rigid and nonrigid transform
using SimpleITK BSplinesTransform are calculated. For convenience, surface models and labels of the umbo displacement
measured by rigid registration of the umbo displacement (project of my supervisor) are already loaded into the dataset
for visualization.
As this is a prototyping project, the output is a pickled xarray. The main advantage of xarray is convenient slicing
along multiple axes as for data cubes and the support of data heterogeneity to the required extent. This output xarray.Dataset
is consumed by the visualization.py script to check the soundness of the TM displacements.

As this script is intended to be run only once on a SLURM cluster or only on change of hyperparameters not set in the
grid values. Since reuseability and maintainability are only of minor interest in this small prototyping project, the
main part of this script runs linearly and is not wrapped into functions.

Please note that the data cannot be made publicly available to run this script.
"""

import sys
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import nrrd
import pyvista as pv
import pickle
import SimpleITK as sitk
import json

#Hardcoded global constants: tbd
ANATOMICAL_FRAME ="" #.npy'
UMBO_DISPLACEMENT_LABELS_PATH = "" #.csv, retrieved column name: 'umbo_cols'
TM_REFERENCE_CURVE_PATH = "" #.json of points on the reference curve (for visualization)
UMBO_REFERENCE_POINT_PATH = "" #.json of the reference umbo point for visualization
TM_VTK_PATH = "" # .vtk surface model of the TM for visualization
UMBO_VTK_PATH = "" # .vtk surface model of the Umbo for visualization
MASK_FIXED_PATH = "" # .nrrd mask for the fixed image for registration (reduces searchspace)
MASK_MOVING_PATH = "" # .nrrd mask for the moving image, can be similar to fixed mask (deplacement-dependent)

# Defined global variables

bbox = dict(x=slice(0,900), y=slice(0,580), z=slice(0,137))
bbox_tuple = tuple(bbox.values())
test_slice = dict(x=slice(None), y=slice(None), z=110)
tuple_test_slice = tuple(test_slice.values())
test_slice_x = dict(x=505, y=slice(None), z=slice(None))
tuple_test_slice_x = tuple(test_slice_x.values())
test_slice_y = dict(x=slice(None), y=360, z=slice(None))
tuple_test_slice_y = tuple(test_slice_y.values())

# 3D roi to crop the volume to box containing the tympanic membrane
neg_origin = (-460,-340,-70)
small_bbox = dict(x=slice(460,575), y=slice(340,410), z=slice(70,136))
small_bbox_tuple = tuple(small_bbox.values())
small_test_slice = dict(x=small_bbox['x'], y=small_bbox['y'], z=test_slice['z'])
small_tuple_test_slice = tuple(small_test_slice.values())
small_test_slice_x = dict(x=test_slice_x['x'], y=small_bbox['y'], z=small_bbox['z'])
small_tuple_test_slice_x = tuple(small_test_slice_x.values())
small_test_slice_y = dict(x=small_bbox['x'], y=test_slice_y['y'], z=small_bbox['z'])
small_tuple_test_slice_y = tuple(small_test_slice_y.values())

# Variables for variants of the transform (for grid search)
#Pressure levels in kPa according to the given data
pressures = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
# Control point spacing for BSplineTransform
CpSpacing = [3, 5, 7, 10, 15]
# Number of iterations for BSplineTransform
n_iterations = [i for i in range(10,161,30)]
# Bounding box for cropping the volume to the TM
offset = np.array([460,340,70])
# Conversion factor pixel to µM
pxToMuM = 100


signalVars = [
    'attenuation', 
    'fused',
    'refraction',
    'diffusion'
    ]

# Keys for the umbo displacement vector components and its magnitude
umbo_cols = ['lateral', 'posterior', 'superior', 'magnitude']


uDispl = xr.DataArray(
        pd.read_csv(UMBO_DISPLACEMENT_LABELS_PATH, usecols=umbo_cols),
        coords={'dP': pressures, 'anatAxes': umbo_cols},
        dims=["dP", "anatAxes"]
    )

datasetsDenoised = []

for s in signalVars:
    pkl_file = open(f'{s}_denoised.pkl', 'rb')
    datasetsDenoised.append(pickle.load(pkl_file))
    pkl_file.close()

# Concatenate all datasets to an Xarray dataset for convenient computation of transforms

data = xr.merge(datasetsDenoised, join='exact')


# Append further data
data["UmboDisplacement"] = uDispl
data.coords["CpSpacing"] = (("CpSpacing"), CpSpacing)
data.coords["Iterations"] = (("Iterations"), n_iterations)

# Load the selected umbo point
file = open(UMBO_REFERENCE_POINT_PATH, 'r')
umbo = json.load(file)
file.close()
data["UmboPoint"] = xr.DataArray(np.array(umbo["markups"][0]["controlPoints"][0]["position"]) - offset,
                           dims=['xyz']) # x: 40, y: 18, z: 40

# Load the curve across the TM, passing through the umbo point above
file = open(TM_REFERENCE_CURVE_PATH, 'r')
tm = json.load(file)
file.close()
data["TM_curve"] = xr.DataArray(np.array([i["position"] for i in tm["markups"][0]["controlPoints"]]) - offset,
                        dims=['nPoints', 'xyz'])

# Load the Basis-Matrix of the anatomical coordinate system and save the inverse as a change-of-basis matrix from the image coordinate system (order: x,y,z (!)) to the anatomical one
anaTransf = np.load(ANATOMICAL_FRAME, allow_pickle = True)
anaTransfMatrix = np.array(list(anaTransf.item().values())[1:])

data["AnaTransfMat"] = xr.DataArray(np.linalg.inv(anaTransfMatrix.T),
                                dims=['rows', 'cols'])

# Load the mesh of the TM and append it as an object to the Dataset; for convenient computation of the displacement magnitudes below
reader = pv.get_reader(TM_VTK_PATH)
mesh = reader.read()
data["mesh"] = mesh

reader = pv.get_reader(UMBO_VTK_PATH)
umboModel = reader.read()
data["umboModel"] = umboModel

# Load moving Image segmentation mask covering the reasonable displacement space of the TM for all static pressures
# the first mask index encodes the saved mask
# The axes are inconsistent with the image arrayys and need to be transposed to the order of array dimensions: rows: x, cols: y, depth: z

mm, _ = nrrd.read(MASK_MOVING_PATH) #
mask_moving = mm[bbox_tuple].transpose(2,1,0)

# Load a smaller mask that roughly segments the TM as the fixed image mask
mf, _ = nrrd.read(MASK_FIXED_PATH) # 
mask_fixed = mf[bbox_tuple].transpose(2,1,0)

# Append the masks to the dataset for convenience

data["mask_moving"] = (('z', 'y', 'x'), mask_moving)
data["mask_fixed"] = (('z', 'y', 'x'), mask_fixed)


def register_euler(fixed, moving, 
        learningRate=0.5,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6):
    """
    Perform 2D rigid image registration using Euler2DTransform.

    Parameters:
    -----------
    fixed : SimpleITK.Image
        The fixed (reference) image.
    moving : SimpleITK.Image
        The moving image to register to fixed.

    Returns:
    --------
    final_transform : SimpleITK.Euler2DTransform
        The computed transform.
    registered : SimpleITK.Image
        The resampled registered moving image.
    """

    # Shift the origin of the full images such that the default origin of the cropped image for BSplineRegistration (0,0,0) coincides with (0,0,0) of the full image and therefore the center of
    # rotation for the rigid transformation is valid for both the full and the cropped volume.

    fixed.SetOrigin(neg_origin)
    moving.SetOrigin(neg_origin)

     # Initial transform centered on image geometry
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Set up registration
    registration = sitk.ImageRegistrationMethod()

    # Metric: Mean Squares (good for mono-modal images)
    registration.SetMetricAsMeanSquares()
    registration.SetMetricSamplingStrategy(registration.NONE) # NONE: use all voxels
    #registration_method.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer: Regular Step Gradient Descent
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate, minStep, numberOfIterations, gradientMagnitudeTolerance,)
    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework. Three-tier pyramid
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration.SetInitialTransform(initial_transform, inPlace=False)

    return registration.Execute(fixed, moving)

# Compute rigid transformations for all pressures in dataset
# We only use the fused images as these provide the best SNR
transforms = []

for p in data.coords["dP"].values:
    print(datetime.datetime.now())
    transform = register_euler(sitk.GetImageFromArray(data['fused_denoised'].sel(dP=0.0)), 
                                  sitk.GetImageFromArray(data['fused_denoised'].sel(dP=p)), 
                                  learningRate=0.3, minStep=5e-7, numberOfIterations=50, gradientMagnitudeTolerance=1e-7)
    
    print(f"\n Rigid Transform for dP {p} calculated.\n")
    transforms.append(transform)
data["rigidTF"] = (("dP"), transforms)

def bspline_registration(
    fixed_image,
    moving_image,
    fixed_image_mask=None,
    moving_image_mask=None,
    fixed_points=None,
    moving_points=None,
    initTrans=None,
    rigid_transform=sitk.Euler3DTransform(),
    cp_spacing = 4,
    spline_order = 2,
    iterations = 6
):
    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    grid_physical_spacing = [cp_spacing for _ in range(0,3)]  # Isotropically spaced control points
    image_physical_size = [
        size * spacing
        for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())
    ]
    mesh_size = [
        int(image_size / grid_spacing + 0.5)
        for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)
    ]
    if initTrans:
        registration_method.SetInitialTransform(initTrans, inPlace = False) # inPlace = False such that ITK internally makes a copy of the transform (not accessible during registration)
    else:
        initial_transform = sitk.BSplineTransformInitializer(
            image1=fixed_image, 
            transformDomainMeshSize=mesh_size, 
            order=spline_order
        )
        registration_method.SetInitialTransform(initial_transform)

    # Sets the rigid transformation (or rather its inverse) to be applied previously to the moving image. This transform is not further optimized.
    registration_method.SetMovingInitialTransform(rigid_transform)

    registration_method.SetMetricAsMeanSquares()
    #registration_method.SetMetricAsANTSNeighborhoodCorrelation(11) # Set radius for NCC patches, kernel size: (2 * r + 1) x NDim 
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(1.00)
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)
    if moving_image_mask:
        registration_method.SetMetricMovingMask(moving_image_mask)


    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1, 1, 1, 1, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    #registration_method.SetOptimizerAsLBFGSB(
    #    gradientConvergenceTolerance=1e-5, numberOfIterations=iterations
    #)
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-10, #1e-5
        numberOfIterations=iterations, #500
        maximumNumberOfCorrections = 35, #5
        maximumNumberOfFunctionEvaluations = 4000, #2000
        costFunctionConvergenceFactor = 1e+5, #  Typical values for factor: 1e+12 for low accuracy; 1e+7 for moderate accuracy and 1e+1 for extremely high accuracy. #default: 1e+7
        lowerBound = -5,
        upperBound = 5
    )
    final_transform = registration_method.Execute(fixed_image, moving_image)

    print(f"Final metric value: {registration_method.GetMetricValue()}")
    print(
        f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
    )
    return final_transform

# Perform registration

bsplines = {
    t : xr.DataArray(
        np.empty((len(data.dP), len(data.CpSpacing), len(data.Iterations)), dtype=object),
        coords=[data.dP, data.CpSpacing, data.Iterations],
        dims=["dP", "CpSpacing", "Iterations"]
    ) for t in signalVars
}
    
for t in signalVars:
    previous =  0
    # The iteration order over dP values was rearranged to first register the volumes 0:3 kPa while each increment is initialized with the previous BSplineTransform for stability and computational efficiency.
    # The volumes -0.5:-3 kPa are registered similarly with the first one initialized with an identity BSplineTransform.
    for p in np.concatenate((data.coords["dP"].values[4:], data.coords["dP"].values[3::-1])):        
        for s in data.coords["CpSpacing"].values:
            for i in data.coords["Iterations"].values:
                print(datetime.datetime.now())
                bsplines[t].loc[dict(dP=p, CpSpacing=s, Iterations=i)] = bspline_registration(
                    sitk.GetImageFromArray(data[f'{t}_denoised'].sel(dP=0.0).isel(small_bbox)), 
                    sitk.GetImageFromArray(data[f'{t}_denoised'].sel(dP=p).isel(small_bbox)),
                    fixed_image_mask=sitk.GetImageFromArray(data["mask_fixed"].isel(small_bbox)), # ..."].isel(small_bbox)
                    moving_image_mask=sitk.GetImageFromArray(data["mask_moving"].isel(small_bbox)),
                    initTrans = None if abs(p) <= 0.5 else bsplines[t].loc[dict(dP=previous, CpSpacing=s, Iterations=i)].item(),
                    rigid_transform=data.rigidTF.sel(dP=p).item().GetNthTransform(0),
                    cp_spacing = s,
                    iterations = int(i) # This value is passed to a function accepting only an uint argument. Therefore casting from numpy.int64 to int is required. 
                    
                )
                previous = p
                
                
                print(f"\n Calculated: \n ImageType: {t} \n dP: {p}\n Controlpoint spacing: {s}\n Number of Iterations for B-Spline Transform: {i} \n\n")
                # Flush the buffer to stdout to monitor progress in the logfile
                sys.stdout.flush()
    data[f"BSplineTx_{t}"] = bsplines[t]

"""
Having computed also the nonrigid sitk.BSplineTransforms, we compute here already the displacement (magnitudes) of our points of interest 
to ease the computational burden for visualization.

"""
# Prepare empty dictionaries of empty DataArrays to be appended to xr.Dataset::data after computation

transTMCurve = {
    t : xr.DataArray(
        np.empty((len(data.dP), len(data.CpSpacing), len(data.Iterations), len(data['TM_curve'].values), len(data.anatAxes))),
        coords=[data.dP, data.CpSpacing, data.Iterations, data.nPoints, data.anatAxes],
        dims=["dP", "CpSpacing", "Iterations", "nPoints", "anatAxes"]
    ) for t in signalVars
}

transUmbo = {
    t : xr.DataArray(
        np.empty((len(data.dP), len(data.CpSpacing), len(data.Iterations), len(data.anatAxes))),
        coords=[data.dP, data.CpSpacing, data.Iterations, data.anatAxes],
        dims=["dP", "CpSpacing", "Iterations", "anatAxes"]
    ) for t in signalVars
}

meshPoints = {
    t : xr.DataArray(
        np.empty((len(data.dP), len(data.CpSpacing), len(data.Iterations), len(data["mesh"].item().points), len(data.anatAxes))),
        coords=[data.dP, data.CpSpacing, data.Iterations, np.arange(len(data["mesh"].item().points)), data.anatAxes],
        dims=["dP", "CpSpacing", "Iterations", "meshPoints", "anatAxes"]
    ) for t in signalVars
}
"""
Calculate all displacment magnitudes resp. vectors for the points of interest. The vectors are already transformed to the anatomical coordinate system:
medial --> lateral
anterior --> posterior
inferior --> superior


Note: The loops through various hyperparameters are similar to the nonrigid transforms to keep the code general. Arguably also the type in signalVars should be introduced as a coordinate
in a future code refinement.
"""

for t in signalVars:
    for p in data.coords["dP"].values:
        for s in data.coords["CpSpacing"].values:
            for i in data.coords["Iterations"].values:
                tx = data[f"BSplineTx_{t}"].sel(dP=p, CpSpacing=s, Iterations=i).item()
                # Transform points on the TM-curve
                for j, v in enumerate(data['TM_curve'].values):
                    tm_vec = data['AnaTransfMat'].values @ (tx.TransformPoint(np.float64(v)) - np.array(v)) * pxToMuM # np.array(3,)
                    transTMCurve[t].loc[{'dP':p, 'CpSpacing':s, 'Iterations': i, 'nPoints': j}] = np.concatenate((tm_vec, np.array([np.linalg.norm(tm_vec)]))) # np.array(4,)
                # Transform the umbo point
                umbo_vec = data['AnaTransfMat'].values @ (tx.TransformPoint(np.float64(data["UmboPoint"].values)) - np.array(data["UmboPoint"].values)) * pxToMuM #np.aaray(3,)
                transUmbo[t].loc[{'dP':p, 'CpSpacing':s, 'Iterations': i}] = np.concatenate((umbo_vec, np.array([np.linalg.norm(umbo_vec)]))) #np.aaray(4,)
                for k, m in enumerate(data["mesh"].item().points):
                    mesh_vec = data['AnaTransfMat'].values @ (tx.TransformPoint(np.float64(m)) - np.array(m)) * pxToMuM #np.aaray(3,)
                    meshPoints[t].loc[{'dP':p, 'CpSpacing':s, 'Iterations': i, 'meshPoints': k}] = np.concatenate((mesh_vec, np.array([np.linalg.norm(mesh_vec)]))) #np.aaray(4,)
    data[f"transTMCurve_{t}"] = transTMCurve[t]
    data[f"transUmbo_{t}"] = transUmbo[t]
    data[f"meshPointDispl_{t}"] = meshPoints[t]





output = open('Denoised_MSQ_dataset_5stepSmoothing.pkl', 'wb')

excluded = signalVars + [s + '_denoised' for s in signalVars]

# Pickle data excluding the images for visualization in a locally run script
pickle.dump(data.drop_vars(excluded), output)
output.close()

