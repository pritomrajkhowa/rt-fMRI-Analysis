#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate the output from realFMRI Scanner

author: Pritom Rajkhowa
last modified: Dec 2019
"""

#--------------------------------------------------
#Receiver Libraries
# python 2/3 compatibility
#--------------------------------------------------

import yaml
import os
import wx
import subprocess, threading

dir_path = os.path.dirname(os.path.realpath(__file__))



from os.path import join
from threading import Thread
import logging
import json
import atexit
import os
import copy 

import numpy as np
import nibabel as nib
import zmq


import subprocess, threading
from Preprocessing import Preprocessor

dir_path = os.path.dirname(os.path.realpath(__file__))


loggerText=''


class logFrame(wx.Frame):
    
    def __init__(self, parent, title):
        
        super(logFrame, self).__init__(parent, title=title)
        
        self.Centre()
        
        self.frame_number = 1
        
        panel = wx.Panel(self)

        sizer = wx.GridBagSizer(5, 5)


        icon1 = wx.StaticBitmap(panel, bitmap=wx.Bitmap('img/pynealLogoBig.png'))
        sizer.Add(icon1, pos=(0, 0), flag=wx.TOP|wx.LEFT|wx.BOTTOM,
            border=15)

        icon2 = wx.StaticBitmap(panel, bitmap=wx.Bitmap('img/logo.jpg'))
        sizer.Add(icon2, pos=(0, 4), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,
            border=5)
        #sizer.Add(icon, pos=(4, 6), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,
        #    border=5)

        line = wx.StaticLine(panel)
        sizer.Add(line, pos=(1, 0), span=(1, 5),
            flag=wx.EXPAND|wx.BOTTOM, border=10)
               
        
        self.tc3 = wx.TextCtrl(panel, wx.ID_ANY, size=(500,500),
                          style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)
        
        
        #sizer.Add(log, 1, wx.ALL|wx.EXPAND, 5)
        sizer.Add(self.tc3, pos=(2, 1), span=(1, 3), flag=wx.ALL|wx.EXPAND)
        


        button5 = wx.Button(panel, label="Cancel")
        button5.Bind(wx.EVT_BUTTON, self.OnQuitApp)
        sizer.Add(button5, pos=(3, 4), span=(1, 1),
        flag=wx.BOTTOM|wx.RIGHT, border=10)
        

        sizer.AddGrowableCol(2)

        panel.SetSizer(sizer)
        sizer.Fit(self)
        self.on_timer()
        
    def on_timer(self):
        global loggerText
        self.tc3.SetValue(loggerText)
        wx.CallLater(1000, self.on_timer)
        
    def OnQuitApp(self, event):
        
        self.Close()






class Command(object):
    
    output=None
    err=None
    
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            #print 'Thread started'
            try :
            	self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,shell=True)
            	self.output, self.err = self.process.communicate()
            	if self.output is not None and self.output.strip()=='':
                	self.output=None
            #print 'Thread finished'
       	    except Exception as e:
       	    	self.output='Memory Error'

            if self.output=='Memory Error':
                self.output='Termination Failed'
                return self.output
        
	
        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            #print 'Terminating process'
            self.process.terminate()
            thread.join()
            self.output='Termination Failed'
        return self.output
 
 
def runRealinement(Total, mat_path, spm_dir, data_dir):
    
    filename=dir_path+"\\tempMat.m"
    #mat_path="C:\\Program Files\\MATLAB\\R2019b\\bin\\"
    #pm_dir="C:\\spm\\spm12\\spm12"
    #data_dir="C:\\test\\matCode\\matlab-spm-scripts-jsh-master\\data\\sub-08"
    #s_fn="C:\\test\\matCode\\matlab-spm-scripts-jsh-master\\data\\sub-08\\anat\\sub-08_T1w.nii"
    #f_fn="C:\\test\\matCode\\matlab-spm-scripts-jsh-master\\data\\sub-08\\func\\sub-08_task-passiveimageviewing_bold.nii"
    file_to_run="\""+mat_path+"matlab.exe"+"\""+" -nodisplay -nosplash -nodesktop -r "+"\"run('"+filename+"');\""
    
    parameters= "Total = "+str(Total)+";\n"+"count=0;\n"+"mat_path = '"+mat_path+"';\n"+ "spm_dir =  '"+spm_dir+"';\n"+"data_dir = '"+data_dir+"';\n"+"stats_dir = ['"+data_dir+"' filesep 'stats'];\n"+"s_fn =  ['"+data_dir+"' filesep 'anat' filesep 'T1w.nii'];\n"+"\nworks_dir_file = [data_dir filesep 'realine_data.txt'];\n"

    file_check_part="\nif ~exist(stats_dir,'dir')"+"\n\tmkdir(stats_dir)"+"\nend"+"\nfwhm = 6;  % mm\n"+"disp('PREPROCESSING')\n"

    inside_if_body="\n\t\tif count==0\n"+"\n\t\t\tfid = fopen( works_dir_file, 'wt' );\n"+"\n\t\t\tfprintf(fid, '%d %d %d %d %d %d\\n', preproc_data.MP(2,:));\n"+"\n\t\t\tfclose(fid);\n"+"\n\t\telse\n"+"\n\t\t\tfid = fopen( works_dir_file, 'a+');"+"\n\t\t\tfprintf(fid, '%d %d %d %d %d %d\\n', preproc_data.MP(2,:));"+"\n\t\t\tfclose(fid);"+"\n\t\tend\n"+"\n\t\tcount = count+1;\n"+"\n\t\tif count > (Total-1)\n"+"\n\t\t\tbreak"+"\n\t\tend"

    if_case ="\n\t\tpreproc_data = spm_standardPreproc_realine(temp_f_fn, s_fn, fwhm, spm_dir);\n"+inside_if_body

    if_body = "\tif isfile(temp_f_fn)\n"+if_case+"\n\telse\n"+"\n\t\tdisp('FILE DO NOT EXIST');"+"\n\tend\n"

    loop_body ="\nwhile 1\n"+"\tdir_name = count+1;\n"+"\ttemp_f_fn = [data_dir filesep 'func' filesep int2str(dir_name) filesep strcat(strcat('passiveimageviewing_bold_',int2str(dir_name)),'.nii')];\n"+if_body+"\nend\ndisp('PREPROCESSING DO NOT EXIST');\nquit;"
 
    mainFile_content = parameters+'\n'+file_check_part+"\n"+loop_body


    writtingFile( filename , mainFile_content )
    #print(dir_path)
    #print(file_to_run)
    command = Command(file_to_run)
    print(command.run(timeout=30))
 

def runPreprocessing(mat_path,pm_dir,data_dir,s_fn,f_fn):
    filename=dir_path+"\\tempMat.m"
    #mat_path="C:\\Program Files\\MATLAB\\R2019b\\bin\\"
    #pm_dir="C:\\spm\\spm12\\spm12"
    #data_dir="C:\\test\\matCode\\matlab-spm-scripts-jsh-master\\data\\sub-08"
    #s_fn="C:\\test\\matCode\\matlab-spm-scripts-jsh-master\\data\\sub-08\\anat\\sub-08_T1w.nii"
    #f_fn="C:\\test\\matCode\\matlab-spm-scripts-jsh-master\\data\\sub-08\\func\\sub-08_task-passiveimageviewing_bold.nii"
    file_to_run="\""+mat_path+"matlab.exe"+"\""+" -nodisplay -nosplash -nodesktop -r "+"\"run('"+filename+"');\""
    part1="\nspm_dir = '"+pm_dir+"';"+"\naddpath(spm_dir);"+"\ns_fn ='"+s_fn+"';"+"\nf_fn ='"+f_fn+"';"+"\nstats_dir = ['"+data_dir+"' filesep 'stats'];"
    part2="\nif ~exist(stats_dir,'dir')"+"\n\tmkdir(stats_dir)"+"\nend"+"\nfwhm = 6;  % mm"+"\ndisp('PREPROCESSING')"+"\n[d, f, e] = fileparts(s_fn);"+"\n[d1, f1, e1] = fileparts(f_fn);"
    part3="\nif exist([d filesep 'rc1' f e], 'file')"+"\n\tdisp('...preproc already done, saving variables...')"+"\n\tpreproc_data = struct;"+"\n\tpreproc_data.forward_transformation = [d filesep 'y_' f e];"+"\n\tpreproc_data.inverse_transformation = [d filesep 'iy_' f e];"+"\n\tpreproc_data.gm_fn = [d filesep 'c1' f e];"+"\n\tpreproc_data.wm_fn = [d filesep 'c2' f e];"+"\n\tpreproc_data.csf_fn = [d filesep 'c3' f e];"+"\n\tpreproc_data.bone_fn = [d filesep 'c4' f e];"+"\n\tpreproc_data.soft_fn = [d filesep 'c5' f e];"+"\n\tpreproc_data.air_fn = [d filesep 'c6' f e];"+"\n\tpreproc_data.rstructural_fn = [d filesep 'r' f e];"+"\n\tpreproc_data.rgm_fn = [d filesep 'rc1' f e];"+"\n\tpreproc_data.rwm_fn = [d filesep 'rc2' f e];"+"\n\tpreproc_data.rcsf_fn = [d filesep 'rc3' f e];"+"\n\tpreproc_data.rbone_fn = [d filesep 'rc4' f e];"+"\n\tpreproc_data.rsoft_fn = [d filesep 'rc5' f e];"+"\n\tpreproc_data.rair_fn = [d filesep 'rc6' f e];"+"\n\tpreproc_data.rfunctional_fn = [d1 filesep 'r' f1 e1];"+"\n\tpreproc_data.srfunctional_fn = [d1 filesep 'sr' f1 e1];"+"\n\tpreproc_data.mp_fn = [d1 filesep 'rp_' f1 '.txt'];"+"\n\tpreproc_data.MP = load(preproc_data.mp_fn);"
    part4="\nelse"+"\n\tdisp('...running preprocessing batch jobs...')"+"\n\tpreproc_data = spm_standardPreproc_realine(f_fn, s_fn, fwhm, spm_dir);"+"\nend"
    part5="\ndisp('Preprocessing done!')"+"\nspm_check_registration(s_fn, [preproc_data.rfunctional_fn ',1'], preproc_data.rgm_fn, preproc_data.rwm_fn, preproc_data.rcsf_fn)"+"\ndisp('Registration done!')\ndelete('tempMat.m'); "
    mainFile_content=part1+part2+part3+part4+part5
    writtingFile( filename , mainFile_content )
    #print(dir_path)
    #print(file_to_run)
    command = Command(file_to_run)
    print(command.run(timeout=30))

########################################################################
#File Operation Start
########################################################################
"""
Reading the contain of the file 
"""
def readingFile( filename ):
    content=None
    with open(filename) as f:
        content = f.readlines()
    return content
 
"""
Wrtitting the contain on file 
"""
def writtingFile( filename , content ):
    try:
        file = open(filename, "w")
        file.write(str(content))
        file.close()
    except IOError:
        print("Error: File does not appear to exist.")
        file.close()


"""
Appending the contain on file 
"""
def appendingFile( filename , content ):
    file = open(filename, "a")
    file.write(str(content))
    file.close()
    
    
########################################################################
#File Operation End
########################################################################

def prepRandomDataset(dims):
    """ Prepare a randomized dataset for use with the simulator

    Build a random dataset of shape dims. Build RAS+ affine (just identiy
    matrix in this case)

    Parameters
    ----------
    dims : list (4 items)
        dimensions of the simulated dataset [x, y, z, t]

    Returns
    -------
    ds : nibabel-like image
        Nibabel dataset

    """
    print('Prepping randomized dataset')
    fakeDataset = np.random.randint(low=0,
                                    high=1,
                                    size=(dims[0],
                                          dims[1],
                                          dims[2],
                                          dims[3]),
                                    dtype='uint16')
    affine = np.eye(4)
    ds = nib.Nifti1Image(fakeDataset, affine)

    print('Randomized Dataset')
    print('Dimensions: {}'.format(ds.shape))
    return ds





class ScanReceiver(Thread):
    """ Class to listen in for incoming scan data.

    As new volumes arrive, the header is decoded, and the volume is added to
    the appropriate place in the 4D data matrix

    Input a dictionary called 'settings' that has (at least) the following keys:
        numTimepts: number of expected timepoints in series [500]
        pynealHost: ip address for the computer running Pyneal
        pynealScannerPort: port # for scanner socket [e.g. 5555]

    """
    def __init__(self, settings):
        """ Initialize the class

        Parameters
        ----------
        settings : dict
            dictionary that contains all of the Pyneal settings for the current
            session. This dictionary is loaded by Pyneal is first launched. At
            a minumum, this dictionary must have the following keys:
                numTimepts: number of expected timepoints in series
                pynealHost: ip address for the computer running Pyneal
                pynealScannerPort: port # for scanner socket [e.g. 5555]

        """
        global loggerText
        
        # start the thread upon creation
        Thread.__init__(self)

        # set up logger
        self.logger = logging.getLogger('PynealLog')

        # get vars from settings dict
        self.numTimepts = settings['numTimepts']
        self.host = settings['pynealHost']
        self.scannerPort = settings['pynealScannerPort']
        self.seriesOutputDir = settings['seriesOutputDir']

        # class config vars
        self.scanStarted = False
        self.firstImage = None
        self.alive = True               # thread status
        self.imageMatrix = None         # matrix that will hold the incoming data
        self.imageMatrixTmp = None         # matrix that will hold the incoming data
        self.affine = None
        self.tr = None

        # array to keep track of completedVols
        self.completedVols = np.zeros(self.numTimepts, dtype=bool)

        # set up socket server to listen for msgs from pyneal-scanner
        self.context = zmq.Context.instance()
        self.scannerSocket = self.context.socket(zmq.PAIR)
        self.scannerSocket.bind('tcp://{}:{}'.format(self.host, self.scannerPort))
        self.logger.debug('bound to {}:{}'.format(self.host, self.scannerPort))
        ### Create processing objects --------------------------
        # Class to handle all preprocessing
        self.preprocessor = Preprocessor(settings)
                
        loggerText='Scan Receiver Server alive and listening....\n'+loggerText
        self.logger.info('Scan Receiver Server alive and listening....')

        # atexit function, shut down server
        atexit.register(self.killServer)



    def run(self):
        global loggerText
        # Once this thread is up and running, confirm that the scanner socket
        # is alive and working before proceeding.
        while True:
            print('Waiting for connection from Scanner')
            loggerText='Waiting for connection from Scanner'+loggerText
            msg = self.scannerSocket.recv_string()
            print('Received message: ', msg)
            loggerText='Received message: '+msg+'\n'+loggerText
            self.scannerSocket.send_string(msg)
            break
        self.logger.debug('scanner socket connected to Scanner')
        loggerText='scanner socket connected to Scanner'+'\n'+loggerText
        #runRealinement(self.numTimepts-1, settings['matLabDir'], settings['spmDir'], settings['seriesOutputDir'])
        # Start the main loop to listen for new data
        while self.alive:
            # wait for json header to appear. The header is assumed to
            # have key:value pairs for:
            # volIdx - volume index (0-based)
            # dtype - dtype of volume voxel array
            # shape - dims of volume voxel array
            # affine - affine to transform vol to RAS+ mm space
            # TR - repetition time of scan
            volHeader = self.scannerSocket.recv_json(flags=0)
            volIdx = volHeader['volIdx']
            self.logger.debug('received volHeader volIdx {}'.format(volIdx));
            loggerText='received volHeader volIdx {}'.format(volIdx)+'\n'+loggerText

            # if this is the first vol, initialize the matrix and store the affine
            if not self.scanStarted:
                self.createImageMatrixStruct(volHeader)
                self.affine = np.array(json.loads(volHeader['affine']))
                self.tr = json.loads(volHeader['TR'])

                self.scanStarted = True     # toggle the scanStarted flag

            if volIdx==-1:
                # now listen for the image data as a string buffer
                voxelArray = self.scannerSocket.recv(flags=0, copy=False, track=False)

                # format the voxel array according to params from the vol header
                voxelArray = np.frombuffer(voxelArray, dtype=volHeader['dtype'])
                voxelArray = voxelArray.reshape(volHeader['shape'])
                
                

                
                # add the volume to the appropriate location in the image matrix
                self.imageMatrix[:, :, :] = voxelArray

                # update the completed volumes table
                self.completedVols[volIdx] = True
                
                # Saving Data in Temp times
                self.saveResultsStruct()

                # send response back to Pyneal-Scanner
                response = 'received volIdx {}'.format(volIdx)
                print('received Structural Image')
                loggerText='received Structural Image'+'\n'+loggerText
                loggerText=response+'\n'+loggerText
                
                self.scannerSocket.send_string(response)
                self.logger.info(response)
                
            else:
                
                ### Set up remaining configuration settings after first volume arrives
                #while not self.completedVols[volIdx]:
                #    time.sleep(.1)
                #preprocessor.set_affine(scanReceiver.get_affine())
                
                if volIdx==0:
                    
                    self.numTimepts = volHeader['numTimepts']
                    self.createImageMatrixFunc(volHeader)
                    self.affine = np.array(json.loads(volHeader['affine']))
                    self.tr = json.loads(volHeader['TR'])
                    self.preprocessor.set_affine(self.affine)
                
                
                # now listen for the image data as a string buffer
                voxelArray = self.scannerSocket.recv(flags=0, copy=False, track=False)

                # format the voxel array according to params from the vol header
                voxelArray = np.frombuffer(voxelArray, dtype=volHeader['dtype'])
                voxelArray = voxelArray.reshape(volHeader['shape'])
                
                
                if self.firstImage is None and volIdx==0:
                    self.firstImage = voxelArray
                    
                self.createImageMatrixTmp(volHeader)
                #self.imageMatrixTmp[:, :, :, 0] = self.firstImage
                self.imageMatrixTmp[:, :, :, 0] = voxelArray
                
                # Saving Data in Temp times
                self.saveResultsFunc(volIdx+1)
                
                preprocVol = self.preprocessor.runPreprocessing(voxelArray, volIdx)

                # add the volume to the appropriate location in the image matrix
                self.imageMatrix[:, :, :, volIdx] = preprocVol

                # update the completed volumes table
                self.completedVols[volIdx] = True
                
                #fun_path = join(self.seriesOutputDir, 'func\\'+str(volIdx) )
                #fun_file_name = 'passiveimageviewing_bold_'+str(volIdx)+'.nii'
                
                #struct_path = join(self.seriesOutputDir, 'anat')
                #struct_file_name = 'T1w.nii'
                
                #runPreprocessing(settings['matLabDir'], settings['spmDir'], settings['seriesOutputDir'],join(fun_path, fun_file_name),join(struct_path, struct_file_name))
                
                # send response back to Pyneal-Scanner
                response = 'received volIdx {} of functional Image'.format(volIdx)
                print(response)
                loggerText=response+'\n'+loggerText
                self.scannerSocket.send_string(response)
                self.logger.info(response)

    
    
    
    def createImageMatrixStruct(self, volHeader):
        """ Create empty 3D image matrix

        Once the first volume appears, this function should be called to build
        the empty matrix to store incoming vol data, using info contained in
        the vol header.

        Parameters
        ----------
        volHeader : dict
            dictionary containing header information from the volume, including
            'volIdx', 'dtype', 'shape', and 'affine'

        """
        # create the empty imageMatrix
        self.imageMatrix = np.zeros(shape=(
            volHeader['shape'][0],
            volHeader['shape'][1],
            volHeader['shape'][2]), dtype=volHeader['dtype'])

        self.logger.debug('Image Matrix dims: {}'.format(self.imageMatrix.shape))

    def createImageMatrixFunc(self, volHeader):
        """ Create empty 4D image matrix

        Once the first volume appears, this function should be called to build
        the empty matrix to store incoming vol data, using info contained in
        the vol header.

        Parameters
        ----------
        volHeader : dict
            dictionary containing header information from the volume, including
            'volIdx', 'dtype', 'shape', and 'affine'

        """
        # create the empty imageMatrix
        self.imageMatrix = np.zeros(shape=(
            volHeader['shape'][0],
            volHeader['shape'][1],
            volHeader['shape'][2],
            self.numTimepts), dtype=volHeader['dtype'])

        self.logger.debug('Image Matrix dims: {}'.format(self.imageMatrix.shape))
        
    def createImageMatrixTmp(self, volHeader):
        """ Create empty 4D image matrix

        Once the first volume appears, this function should be called to build
        the empty matrix to store incoming vol data, using info contained in
        the vol header.

        Parameters
        ----------
        volHeader : dict
            dictionary containing header information from the volume, including
            'volIdx', 'dtype', 'shape', and 'affine'

        """
        # create the empty imageMatrix
        self.imageMatrixTmp = np.zeros(shape=(
            volHeader['shape'][0],
            volHeader['shape'][1],
            volHeader['shape'][2],
            1), dtype=volHeader['dtype'])

        self.logger.debug('Image Matrix dims: {}'.format(self.imageMatrix.shape))

    def get_affine(self):
        """ Return the affine for the current series

        """
        return self.affine

    def get_vol(self, volIdx):
        """ Return the requested vol, if it is here.

        Parameters
        ----------
        volIdx : int
            index location (0-based) of the volume you'd like to retrieve

        Returns
        -------
        numpy-array or None
            3D array of voxel data for the requested volume

        """
        if self.completedVols[volIdx]:
            return self.imageMatrix[:, :, :, volIdx]
        else:
            return None

    def get_slice(self, volIdx, sliceIdx):
        """ Return the requested slice, if it is here.

        Parameters
        ----------
        volIdx : int
            index location (0-based) of the volume you'd like to retrieve
        sliceIdx : int
            index location (0-based) of the slice you'd like to retrieve

        Returns
        -------
        numpy-array or None
            2D array of voxel data for the requested slice

        """
        if self.completedVols[volIdx]:
            return self.imageMatrix[:, :, sliceIdx, volIdx]
        else:
            return None


    def saveResults(self):
        """ Save the numpy 4D image matrix of data as a Nifti File

        Save the image matrix as a Nifti file in the output directory for this
        series

        """
        # build nifti image
        ds = nib.Nifti1Image(self.imageMatrix, self.affine)

        # set the TR appropriately in the header
        pixDims = np.array(ds.header.get_zooms())
        pixDims[3] = self.tr
        ds.header.set_zooms(pixDims)

        # save to disk
        nib.save(ds, join(self.seriesOutputDir,  'receivedFunc.nii.gz'))
        
    def saveResultsStruct(self):
        """ Save the numpy 3D image matrix of data as a Nifti File

        Save the image matrix as a Nifti file in the output directory for this
        series

        """
        
        path = join(dir_path,'temp', 'anat')
        file_name = 'T1w.nii'
        #Creat Path if not exits
        self.creatDirectory(path)
        # build nifti image
        ds = nib.Nifti1Image(self.imageMatrix, self.affine)

        # set the TR appropriately in the header
        pixDims = np.array(ds.header.get_zooms())
        ds.header.set_zooms(pixDims)

        # save to disk
        nib.save(ds, join(path, file_name))
        
    def saveResultsFunc(self, volIdx):
        """ Save the numpy 4D image matrix of data as a Nifti File

        Save the image matrix as a Nifti file in the output directory for this
        series

        """
        
        path = join(dir_path, 'temp', 'func/'+str(volIdx) )
        file_name = 'passiveimageviewing_bold_'+str(volIdx)+'.nii'
        #Creat Path if not exits
        self.creatDirectory(path)
        
        # build nifti image
        ds = nib.Nifti1Image(self.imageMatrixTmp, self.affine)

        # set the TR appropriately in the header
        pixDims = np.array(ds.header.get_zooms())
        pixDims[3] = self.tr
        ds.header.set_zooms(pixDims)
        # save to disk
        nib.save(ds, join(path, file_name))

    def saveImageTemp(self, tempImage):
        """ Save the numpy 4D image matrix of data as a Nifti File

        Save the image matrix as a Nifti file in the output directory for this
        series

        """
        path = join(dir_path, 'temp')
        file_name = 'tempFile.nii'
        # set the TR appropriately in the header
        pixDims = np.array(tempImage.header.get_zooms())
        pixDims[3] = self.tr
        tempImage.header.set_zooms(pixDims)
        # save to disk
        nib.save(tempImage, join(path, file_name))



    def killServer(self):
        """ Close the thread by setting the alive flag to False """
        self.context.destroy()
        self.alive = False
    
    def creatDirectory(self, path):
        try:
            os.makedirs(path)
            print ("Successfully created the directory %s" % path)
                
        except OSError:
            print ("Creation of the directory %s failed" % path)





########################################################################
#File Operation End
########################################################################


wildcard = "NII source (*.nii)|*.nii|" \
            "All files (*.*)|*.*" 

class Example(wx.Frame):

    def __init__(self, parent, title):
        super(Example, self).__init__(parent, title=title)
        self.currentDirectory = os.getcwd()
        self.input_spm_path_file=None
        self.input_func_file=None
        self.input_struct_file=None
        self.logInfo='Test\n'
        self.InitUI()
        self.Centre()
        self.frame_number = 1

    def InitUI(self):

        panel = wx.Panel(self)

        sizer = wx.GridBagSizer(5, 5)


        icon1 = wx.StaticBitmap(panel, bitmap=wx.Bitmap('img/pynealLogoBig.png'))
        sizer.Add(icon1, pos=(0, 0), flag=wx.TOP|wx.LEFT|wx.BOTTOM,
            border=15)

        icon2 = wx.StaticBitmap(panel, bitmap=wx.Bitmap('img/logo.jpg'))
        sizer.Add(icon2, pos=(0, 4), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,
            border=5)
        #sizer.Add(icon, pos=(4, 6), flag=wx.TOP|wx.RIGHT|wx.ALIGN_RIGHT,
        #    border=5)


        line = wx.StaticLine(panel)
        sizer.Add(line, pos=(1, 0), span=(1, 5),
            flag=wx.EXPAND|wx.BOTTOM, border=10)
        
        text1 = wx.StaticText(panel, label="Event Details")
        sizer.Add(text1, pos=(2, 0), flag=wx.LEFT, border=10)

        self.tc1 = wx.TextCtrl(panel)
        self.tc1.SetValue("/home/pritom/nistates/update/temp/file_event.tsv")
        sizer.Add(self.tc1, pos=(2, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND|wx.TE_READONLY)

        text2 = wx.StaticText(panel, label="Activation for the Condition")
        sizer.Add(text2, pos=(3, 0), flag=wx.LEFT, border=10)

        self.tc2 = wx.TextCtrl(panel)
        self.tc2.SetValue("food")
        sizer.Add(self.tc2, pos=(3, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND|wx.TE_READONLY)
        
        
        text3 = wx.StaticText(panel, label="Number Of Time Point")
        sizer.Add(text3, pos=(4, 0), flag=wx.LEFT, border=10)

        self.tc3 = wx.TextCtrl(panel)
        self.tc3.SetValue("375")
        sizer.Add(self.tc3, pos=(4, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND)
        
        text4 = wx.StaticText(panel, label="Scanner Receiver IP")
        sizer.Add(text4, pos=(5, 0), flag=wx.LEFT, border=10)

        self.tc4 = wx.TextCtrl(panel)
        self.tc4.SetValue("127.0.0.1")
        sizer.Add(self.tc4, pos=(5, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND|wx.TE_READONLY)
      
        
        text5 = wx.StaticText(panel, label="Scanner Port")
        sizer.Add(text5, pos=(6, 0), flag=wx.LEFT, border=10)

        self.tc5 = wx.TextCtrl(panel)
        self.tc5.SetValue("5555")
        sizer.Add(self.tc5, pos=(6, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND|wx.TE_READONLY)
        
        
        button3 = wx.Button(panel, label='Help')
        sizer.Add(button3, pos=(9, 0), flag=wx.LEFT, border=10)

        button4 = wx.Button(panel, label="Start")
        button4.Bind(wx.EVT_BUTTON, self.OnOkApp)
        sizer.Add(button4, pos=(9, 3))

        button5 = wx.Button(panel, label="Cancel")
        button5.Bind(wx.EVT_BUTTON, self.OnQuitApp)
        sizer.Add(button5, pos=(9, 4), span=(1, 1),
            flag=wx.BOTTOM|wx.RIGHT, border=10)

        sizer.AddGrowableCol(2)

        panel.SetSizer(sizer)
        sizer.Fit(self)
        
        #self.on_timer()
        
   


        
        
    #----------------------------------------------------------------------
    def onFunOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            #print(paths)
            #print("You chose the following file(s):")
            for path in paths:
                self.input_func_file = path
                self.tc4.SetValue(self.input_func_file)
                #print(path)
        #print(self.input_func_file)
        dlg.Destroy()
        
    def onStructOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            #print(paths)
            #print("You chose the following file(s):")
            for path in paths:
                self.input_struct_file = path
                self.tc5.SetValue(self.input_struct_file)
                #print(path)
        #print(self.input_struct_file)
        dlg.Destroy()
        
    def OnOkApp(self, event):
        
        dict_file={}
        
        matlabFile=self.tc1.GetValue()
        dict_file['math_path'] = matlabFile
        
        spmfile=self.tc2.GetValue()
        dict_file['spm_dir'] = spmfile
        
        numberofTimept=self.tc3.GetValue()
        dict_file['numberofTimept'] = numberofTimept
        
        ipaddress=self.tc4.GetValue()
        dict_file['receiver_ipaddress'] = ipaddress
        
        socket_host=self.tc5.GetValue()
        dict_file['receiver_socket'] = socket_host
        
        setting_file=dir_path+'/'+'setting.yaml'
        
        with open(setting_file, 'w') as file:
            
            documents = yaml.dump(dict_file, file)

            
        self.Close()
        #print(command.run(timeout=30))
        #self.Close()
       

    def OnQuitApp(self, event):
        self.Close()
        
    def onChecked(self, e): 
      cb = e.GetEventObject() 
      #print(cb.GetLabel(),' is clicked',cb.GetValue())
      
      
def startReceiver():
    
    setting_file=dir_path+'/'+'setting.yaml'
    
    with open(setting_file) as file:
        
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        
        parameter_list = yaml.load(file, Loader=yaml.FullLoader)
        
        if parameter_list is not None:
            
            math_path = parameter_list['math_path'] 
            
            spm_dir = parameter_list['spm_dir'] 
            
            numberofTimept = parameter_list['numberofTimept']
            
            receiver_ipaddress = parameter_list['receiver_ipaddress']
            
            receiver_socket = parameter_list['receiver_socket']
            
            if isValidParameter(math_path) != None and  isValidParameter(spm_dir) != None and isValidParameter(numberofTimept) != None and isValidParameter(receiver_ipaddress) != None and isValidParameter(receiver_socket) != None:
                            
                settings = {'numTimepts': int(numberofTimept), 'pynealHost':receiver_ipaddress,
                'pynealScannerPort': int(receiver_socket), 'seriesOutputDir':dir_path+"/temp01/", 'matLabDir':math_path,'spmDir':spm_dir,'launchDashboard':False,'estimateMotion':True}

                ### set up logging
                fileLogger = logging.FileHandler('./scanReceiver.log', mode='w')
                fileLogger.setLevel(logging.DEBUG)
                fileLogFormat = logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(module)s, line: %(lineno)d - %(message)s',
                                      '%m-%d %H:%M:%S')
                fileLogger.setFormatter(fileLogFormat)
                logger = logging.getLogger()
                logger.setLevel(logging.DEBUG)
                logger.addHandler(fileLogger)
                

                
                # start the scanReceiver
                scanReceiver = ScanReceiver(settings)
                scanReceiver.start()
                



def isValidParameter(stringValue):
     
    if stringValue is not None and stringValue.strip()!='':
       return stringValue
    else:
       return None


def main():

    app = wx.App()
    ex = Example(None, title="Scanner Image Processing Module")
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
    app = wx.App()
    startReceiver()
    frame = logFrame(None, title="Scanner Image Processing Module Log")
    frame.Show()
    app.MainLoop()
