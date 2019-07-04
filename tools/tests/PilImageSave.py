from PIL import Image
import numpy as np

imgPath = '/var/www/webdav/Data/ICA/Scenes/Validation/Scene25/renders/Segmentations/00253.png'
targetPath = '/home/comvis/Temp/hurr.png'
imgFile = Image.open(imgPath)
imgFileWEB = imgFile.convert('P', palette=Image.WEB)
webPalette = imgFileWEB.getpalette()
data = np.array(imgFileWEB)
data = data*2
data[data==4] = 0
data[data==10] = 0
imgFileNew = Image.fromarray(data)
imgFileNew.putpalette(webPalette)
imgFileNew.save(targetPath)

# segImg = np.asarray(imgFile)
# print(segImg.shape)
# print(np.unique(segImg))
# j = Image.fromarray(segImg)
# j.save(targetPath)
# segImg2 = np.asarray(Image.open(targetPath))
# print(segImg2.shape)
# print(np.unique(segImg2))
# print(np.sum((segImg-segImg2)==0))
# print(segImg.shape[0]*segImg.shape[1])
