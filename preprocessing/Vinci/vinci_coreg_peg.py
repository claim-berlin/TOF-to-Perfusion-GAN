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
out_path = "/data/PEGASUS/"
perfMap = sys.argv[2]
patNum = int(sys.argv[1])


files = os.listdir(top_path)
pat_ids = []

for file in files:
    tmp = file.split("_")[0]
    if tmp not in pat_ids:
        pat_ids.append(tmp)

target_shape = (256,256,127)

pat_ids = ["PEG0003", "PEG0004", "PEG0005", "PEG0009", "PEG0010", "PEG0011",
             "PEG0013", "PEG0018", "PEG0019", "PEG0020", "PEG0021", "PEG0022", 
             "PEG0023", "PEG0027", "PEG0028", "PEG0034", "PEG0038", "PEG0043", 
             "PEG0044", "PEG0006", "PEG0008", "PEG0014", "PEG0015", "PEG0016", 
             "PEG0017", "PEG0024", "PEG0025", "PEG0026", "PEG0029", "PEG0030", 
             "PEG0031", "PEG0032", "PEG0033", "PEG0035", "PEG0036", "PEG0037",
             "PEG0040", "PEG0041", "PEG0042", "PEG0045", "PEG0046", "PEG0047", 
             "PEG0048", "PEG0049", "PEG0050", "PEG0052", "PEG0053", "PEG0054", 
             "PEG0055", "PEG0056", "PEG0057", "PEG0058", "PEG0059", "PEG0060", 
             "PEG0061", "PEG0062", "PEG0063", "PEG0064", "PEG0065", "PEG0066", 
             "PEG0067", "PEG0068", "PEG0069", "PEG0070", "PEG0071", "PEG0072", 
             "PEG0073", "PEG0074", "PEG0075", "PEG0076", "PEG0078", "PEG0079"]

#pat_ids = pat_ids[patNum:]
# generate param xmls for each patient -> 
for pat_id in pat_ids:
    print("Start")
    szJobSummaryXML = top_path + pat_id + "_coreg_results"
    szMMMTemplateFile = "peg_params.xml" # parameters for registration job
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
    print("End")

if not bStayOpen:
	rsl.killYourself()	#close image buffer
	ref.killYourself()

if not bStayOpen:
	con.CloseVinci(True)
else:
	con.CloseSockets()

