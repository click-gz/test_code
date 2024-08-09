import SimpleITK as sitk
import numpy  as np
import matplotlib.pyplot as plt
img = sitk.GetArrayFromImage(sitk.ReadImage("./out.nii"))
img = (np.max(img) - img)/(np.max(img) - np.min(img))
sitk.WriteImage(sitk.GetImageFromArray(img), "./out.nii")
img = sitk.GetArrayFromImage(sitk.ReadImage("./out_cube.nii"))
img = (np.max(img) - img)/(np.max(img) - np.min(img))
sitk.WriteImage(sitk.GetImageFromArray(img), "./out_cube.nii")
exit()
fimg = sitk.GetArrayFromImage(sitk.ReadImage("./fixed.nii.gz"))
# fimg = (fimg - np.min(fimg))/(np.max(fimg) - np.min(fimg))
plt.imshow(fimg[:, 95, :], cmap='gray')
plt.colorbar()  # 显示颜色条
plt.show()

slice1 = img[:, 95, :]
img = fimg-img
# sitk.WriteImage(sitk.GetImageFromArray(img), "./out_cd.nii")
img = (img - np.min(img))/(np.max(img) - np.min(img))

slice = img[:, 95, :]
# slice[slice<0.4] = 0.1
plt.imshow(slice1, cmap='gray')
plt.colorbar()  # 显示颜色条
plt.show()
plt.imshow(slice, cmap = 'viridis')

plt.colorbar()  # 显示颜色条
plt.show()

