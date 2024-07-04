#!/usr/bin/env python
# -*- coding: latin-1 -*-
# $Id: RunMMMJob.py 18607 2018-01-15 12:30:18Z michael $

"""
This file is part of the Vinci 4.x distribution and used 
for communication with Vinci from a remote Python program.
See ReadMe.txt for further information.

(c) Max-Planck-Institute for 
Metabolism Research (MPIfSF)
Cologne, Germany 2005-2018
http://vinci.sf.mpg.de
Email: vinci3@sf.mpg.de

Best viewed with tabs set to 4 spaces.

This is a Python program illustrating how to load an MRI image volume
and a PET image volume into Vinci and have Vinci's MMM plugin
register the PET volume to the MRI data.
The registration result file is saved in an XML file.

The file "VinciPy/Vinci_Py.xml" defines which version of Vinci will be used.
"""

import sys
import os
if sys.version.startswith("3"):
	from VinciPy3 import *
else:
	from VinciPy import *


#print("This is RunMMMJob 0.41 of August 5, 2013.")
#print("  uses images in the vinci_demo_data directory")
#print("  which should be installed in parallel to vinci_xxx")

bStayOpen=len(sys.argv) > 1 and sys.argv[1] == "-o"

bin=Vinci_Bin.Vinci_Bin()
con=Vinci_Connect.Vinci_Connect(bin)
szVinciBinPath=con.StartMyVinci()

vc=Vinci_Core.Vinci_CoreCalc(con)
vc.StdProject()

top_path = "/data/PEGASUS/unprocessed/"
out_path = "/data/PEGASUS/processed/skullstripping/"
perfMap = sys.argv[1]


files = os.listdir(top_path)
pat_ids = []

for file in files:
    tmp = file.split("_")[0]
    if tmp not in pat_ids:
        pat_ids.append(tmp)

target_shape = (256,256,98)
pat_ids = ['0027', '0029', '0034', '0046', '0049', '0059', '0062', '0071', '0074', '0075', '0083', 
'0088', '0092', '0096', '0105', '0124', '0129', '0134', '0136', '0162', '0170', '0178', '0179', '0180', 
'0184', '0188', '0194', '0195', '0196', '0197', '0199', '0202', '0208', '0209', '0210', '0211', '0214', '0215', 
'0216', '0218', '0219', '0220', '0228', '0233', '0237', '0240', '0241', '0242', '0245', '0255', '0258', '0263', '0266', 
'0267', '0270', '0272', '0273', '0275', '0277', '0287', '0288', '0290', '0293', '0299', '0302', '0305', '0306', '0307', '0308', 
'0310', '0313', '0314', '0317', '0318', '0320', '0321', '0325', '0329', '0331', '0332', '0333', '0334', '0335', '0337', '0342', 
'0344', '0345', '0346', '0348', '0350', '0352', '0355', '0356', '0357', '0360', '0362', '0363', '0365', '0367', '0368', '0371', 
'0373', '0378', '0380', '0382', '0383', '0388', '0390', '0391', '0393', '0394', '0395', '0396', '0397', '0400', '0402', '0403', 
'0407', '0410', '0417', '0420', '0421', '0430', '0432', '0433', '0435', '0442', '0446', '0451', '0460', '0464', '0465', '0466', 
'0473', '0483', '0488', '0492', '0494', '0495', '0496', '0513', '0515', '0519', '0528', '0534', '0547', '0565', '0574', '0591', 
'0604', '0638', '0639', '0661', '0668', '0676', '0681', '0686', '0690', '0695', '0704', '0705', '0718', '0720', '0729', '0731', 
'0764', '0766', '0767', '0774', '0785', '0792', '0826', '0835', '0839', '0843', '0887', '0907', '0908', '0910', '0933', '0960', 
'0979', '1015', '1024', '1029', '1038', '1081', '1112', '1139', '1143', '1153', '1169', '1182', '1213', '1362', '1414', '1491', '1508', '1514', '1552']


# generate param xmls for each patient -> 
for pat_id in pat_ids:
    szJobSummaryXML = top_path + pat_id + "_coreg_results"
    szMMMTemplateFile = "hdb_params.xml" # parameters for registration job
    szReferenceImage = top_path + pat_id + "_TOF.nii.gz"
    szReslicingImage = top_path + pat_id + "_DSC_source0.nii.gz"
    szReslicingImage2 = top_path + pat_id + "_" + perfMap + ".nii.gz"

    # output
    szReferenceImageOut = out_path + pat_id + "_TOF.nii.gz"
    szReslicingImageOut = out_path + pat_id + "_DSC_source0.nii.gz"
    szReslicingImage2Out = out_path + pat_id + "_" + perfMap + ".nii.gz"

    f=open(szMMMTemplateFile,"rb")
    szSchemeFile=f.read()
    f.close()

    #check scheme data file and remove possible XML comments at its beginning
    root=Vinci_XML.ElementTree.fromstring(szSchemeFile)
    if root.tag != "MMM":
            sys.exit("scheme data file %s does not contain tag MMM\n"%szMMMTemplateFile)
    szSchemeFile=Vinci_XML.ElementTree.tostring(root)
    
    # bytes -> str
    szSchemeFile=szSchemeFile.decode("utf-8")

    ref=Vinci_ImageT.newTemporary(vc,szFileName=szReferenceImage)
    rsl=Vinci_ImageT.newTemporary(vc,szFileName=szReslicingImage)
    rsl.setColorSettings(CTable="Rainbow4")
    
    # align rsl to ref by using szSchemeFile (params pre generated -> must be DSC to TOF)
    rsl.alignToRef(ref,szSchemeFile,szRegistrationSummaryFile=szJobSummaryXML) # generate result file xml
    #rsl.saveYourselfAs(bUseOffsetRotation=True,szFullFileName= szReslicingImageOut, dimension=[ref.iDim_x,ref.iDim_y,ref.iDim_z],pixelsize=[ref.fPixelSize_x,ref.fPixelSize_y,ref.fPixelSize_z])	#save the registrated image (image.v -> image_rsl.v)
    #rsl.saveYourselfAs(bUseOffsetRotation=True,szFullFileName= szReslicingImageOut, dimension=[target_shape[0], target_shape[1], target_shape[2]], pixelsize=[ref.fPixelSize_x*ref.iDim_x/target_shape[0], ref.fPixelSize_y*ref.iDim_y/target_shape[1], ref.fPixelSize_z*ref.iDim_z/target_shape[2]])
    a = Vinci_ImageT.newTemporary(vc,szFileName=szReslicingImage2)
    a.setColorSettings(CTable="Rainbow4")

    # align rsl2 to rsl by using generated result file 
    rsl2 = a.reapplyMMMTransform(szJobSummaryXML,IsComputed=True)
    #rsl2.saveYourselfAs(bUseOffsetRotation=True, szFullFileName= szReslicingImage2Out, szNameAppend="_rsl.nii")	

    # save rescliced img with geometry of ref
    #rsl2.saveYourselfAs(bUseOffsetRotation=True,szFullFileName= szReslicingImage2Out,dimension=[ref.iDim_x,ref.iDim_y,ref.iDim_z],pixelsize=[ref.fPixelSize_x,ref.fPixelSize_y,ref.fPixelSize_z])
    rsl2.saveYourselfAs(bUseOffsetRotation=True,szFullFileName= szReslicingImage2Out, dimension=[target_shape[0], target_shape[1], target_shape[2]], pixelsize=[ref.fPixelSize_x*ref.iDim_x/target_shape[0], ref.fPixelSize_y*ref.iDim_y/target_shape[1], ref.fPixelSize_z*ref.iDim_z/target_shape[2]])
    # save ref for positional match in ITK SNAP
    #ref.saveYourselfAs(bUseOffsetRotation=True,szFullFileName= szReferenceImageOut)
    #ref.saveYourselfAs(bUseOffsetRotation=True,szFullFileName= szReferenceImageOut, dimension=[target_shape[0], target_shape[1], target_shape[2]], pixelsize=[ref.fPixelSize_x*ref.iDim_x/target_shape[0], ref.fPixelSize_y*ref.iDim_y/target_shape[1], ref.fPixelSize_z*ref.iDim_z/target_shape[2]])
if not bStayOpen:
	rsl.killYourself()	#close image buffer
	ref.killYourself()

if not bStayOpen:
	con.CloseVinci(True)
else:
	con.CloseSockets()

