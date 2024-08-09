import SimpleITK as sitk

img = sitk.ReadImage("field.nii")
arry = sitk.GetArrayFromImage(img)
print(arry.shape)