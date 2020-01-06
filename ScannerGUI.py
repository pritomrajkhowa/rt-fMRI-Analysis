#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate the output from realFMRI Scanner

author: Pritom Rajkhowa
last modified: Dec 2019
"""

#--------------------------------------------------
#Scanner Libraries
# python 2/3 compatibility
from __future__ import print_function
from __future__ import division
from builtins import input
import yaml
import random
import time
import json
import argparse
import sys
import zmq
import numpy as np
import nibabel as nib
import pandas as pd

#--------------------------------------------------


import os
import wx
import subprocess, threading

dir_path = os.path.dirname(os.path.realpath(__file__))

#Global Variable 

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






def prepRealDataset(struct_image_path, functional_image_path, eventfile_path):
    """ Prepare a real, existing dataset for use with the simulator

    Read in the supplied 4d image file, set orientation to RAS+

    Parameters
    ----------
    struct_image_path : string
        full path to the structural dataset you want to use

    functional_image_path : string
        full path to the functional dataset you want to use
        
    Returns
    -------
    ds_RAS : nibabel-like image
        Nibabel dataset with orientation set to RAS+

    """
    print('Prepping dataset: {}'.format(struct_image_path))
    sds = nib.load(struct_image_path)
    
    print('Prepping dataset: {}'.format(functional_image_path))
    fds = nib.load(functional_image_path)

    # make sure it's RAS+
    sds_RAS = nib.as_closest_canonical(sds)
    
    # make sure it's RAS+
    fds_RAS = nib.as_closest_canonical(fds)

    events = pd.read_table(eventfile_path)
    
    print('Structural Dimensions: {}'.format(sds_RAS.shape))
    print('Functional Dimensions: {}'.format(fds_RAS.shape))
    print('Events Details:')
    print(events)
    return sds_RAS,fds_RAS,events




def scannerSimulator(sdataset, fdataset, eventsdataset, TR=1600, host='127.0.0.1', port=5555):
    """ Scanner Simulator

    Simulate Scanner by sending the supplied dataset to Receiver via
    socket one volume at a time. Rate set by 'TR' argument. Each volume
    preceded with a json header with metadata about volume, just like during a
    real scan

    Paramters
    ---------
    sdataset : nibabel-like image
        Nibabel like image representing the structural dataset you'd like to use for the
        simulation
    sdataset : nibabel-like image
        Nibabel like image representing the functional dataset you'd like to use for the
        simulation
    TR : int, optional
        TR to send the data at. In ms (default: 1000)
    host : string, optional
        Host IP address of Pyneal server. Pyneal Scanner will send data to this
        address (default: '127.0.0.1')
    port : int
        Port number to use for sending data to Pyneal

    """
    global loggerText
    print('TR: {}'.format(TR))
    loggerText='TR: {}'.format(TR)+"\n"+loggerText
    # convert TR to sec (the unit of time.sleep())
    TR = TR / 1000

    # Create socket, bind to address
    print('Connecting to Scaner Server at {}:{}'.format(host, port))
    loggerText='Connecting to Scaner Server at {}:{}'.format(host, port)+"\n"+loggerText
    context = zmq.Context.instance()
    socket = context.socket(zmq.PAIR)
    socket.connect('tcp://{}:{}'.format(host, port))

    sds_array = sdataset.get_data()
    sds_affine = sdataset.affine
    
    fds_array = fdataset.get_data()
    fds_affine = fdataset.affine

    # Wait for pyneal to connect to the socket
    print('waiting for connection...')
    loggerText+='waiting for connection...'+"\n"+loggerText
    while True:
        msg = 'hello from Scanner Simulator'
        socket.send_string(msg)

        resp = socket.recv_string()
        if resp == msg:
            print('connected to Server')
            loggerText+='connected to Server...'+"\n"+loggerText
            break

    # Press Enter to start sending data
    #input('Press ENTER to begin the "scan" ')
    #dlg = wx.MessageDialog(None, "Do you want to  begin the \"scan\" ?",'Scan Starter',wx.YES_NO | wx.ICON_QUESTION)
    #result = dlg.ShowModal()

    #if result == wx.ID_YES:
    #   print("Scan Started")
    #   loggerText+="Scan Started\n"
    #else:
    #   context.destroy()

    # sleep for 1TR to account for first volume being collected
    time.sleep(TR)
    
    volIdx =-1
    
    # grab this volume from the dataset
    thisVol = np.ascontiguousarray(sds_array)

    # build header
    volHeader = {'volIdx': volIdx,
                     'dtype': str(thisVol.dtype),
                     'shape': thisVol.shape,
                     'affine': json.dumps(sds_affine.tolist()),
                     'TR': str(TR*1000)}

    # send header as json
    socket.send_json(volHeader, zmq.SNDMORE)

    # now send the voxel array for this volume
    socket.send(thisVol, flags=0, copy=False, track=False)
    print('Sent Structural image')
    
    loggerText='Sent Structural image\n'+loggerText

    # list for response
    socketResponse = socket.recv_string()
    print('Socket Response: {}'.format(socketResponse))
    loggerText='Socket Response: {}'.format(socketResponse)+loggerText

    
    startTime = time.time()
    
    if TR > 0:
        elapsedTime = time.time() - startTime
        time.sleep(TR - elapsedTime)


    # Start sending data!
    for volIdx in range(fds_array.shape[3]):
        
        
        startTime = time.time()

        # grab this volume from the dataset
        thisVol = np.ascontiguousarray(fds_array[:, :, :, volIdx])

        # build header
        volHeader = {'volIdx': volIdx,
                     'dtype': str(thisVol.dtype),
                     'shape': thisVol.shape,
                     'affine': json.dumps(fds_affine.tolist()),
                     'TR': str(TR*5000), 'numTimepts':fds_array.shape[3]}

        # send header as json
        socket.send_json(volHeader, zmq.SNDMORE)

        # now send the voxel array for this volume
        socket.send(thisVol, flags=0, copy=False, track=False)
        print('Sent vol: {}'.format(volIdx))
        loggerText='Sent vol: {}'.format(volIdx)+"\n"+loggerText


        # list for response
        socketResponse = socket.recv_string()
        print('Socket Response: {}'.format(socketResponse))
        loggerText='Socket Response: {}'.format(socketResponse)+"\n"+loggerText

        if TR > 0:
            elapsedTime = time.time() - startTime
            
            if(TR<=elapsedTime):
            
                time.sleep(TR)
                
            else:
                
                time.sleep(TR - elapsedTime)
                

    # close the socket
    context.destroy()







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


wildcard = "NII source (*.nii)|*.nii|" \
            "All files (*.*)|*.*" 

wildcard1 = "tsv source (*.tsv)|*.tsv|" \
            "All files (*.*)|*.*" 

class Example(wx.Frame):

    def __init__(self, parent, title):
        super(Example, self).__init__(parent, title=title)
        self.currentDirectory = os.getcwd()
        self.input_spm_path_file=None
        self.input_func_file=None
        self.input_struct_file=None
        self.input_event_file=None
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
        
        text1 = wx.StaticText(panel, label="Scanner Receiver IP")
        sizer.Add(text1, pos=(2, 0), flag=wx.LEFT, border=10)

        self.tc1 = wx.TextCtrl(panel)
        self.tc1.SetValue("127.0.0.1")
        sizer.Add(self.tc1, pos=(2, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND|wx.TE_READONLY)

        text2 = wx.StaticText(panel, label="Scanner Port")
        sizer.Add(text2, pos=(3, 0), flag=wx.LEFT, border=10)

        self.tc2 = wx.TextCtrl(panel)
        self.tc2.SetValue("5555")
        sizer.Add(self.tc2, pos=(3, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND|wx.TE_READONLY)
        
        
             

        text3 = wx.StaticText(panel, label="Functional Image")
        sizer.Add(text3, pos=(4, 0), flag=wx.LEFT|wx.TOP, border=10)

        self.tc4 = wx.TextCtrl(panel,-1,size=(300, -1))
        sizer.Add(self.tc4, pos=(4, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND,
            border=5)

        button1 = wx.Button(panel, label="Browse...")
        button1.Bind(wx.EVT_BUTTON, self.onFunOpenFile)
        
        sizer.Add(button1, pos=(4, 4), flag=wx.TOP|wx.RIGHT, border=5)

        text4 = wx.StaticText(panel, label="Structural Image")
        sizer.Add(text4, pos=(5, 0), flag=wx.TOP|wx.LEFT, border=10)

        self.tc5 = wx.TextCtrl(panel)
        sizer.Add(self.tc5, pos=(5, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND,
            border=5)
        #combo = wx.ComboBox(panel)
        #sizer.Add(combo, pos=(4, 1), span=(1, 3),
        #    flag=wx.TOP|wx.EXPAND, border=5)

        button2 = wx.Button(panel, label="Browse...")
        button2.Bind(wx.EVT_BUTTON, self.onStructOpenFile)
        sizer.Add(button2, pos=(5, 4), flag=wx.TOP|wx.RIGHT, border=5)

        text5 = wx.StaticText(panel, label="Event File")
        sizer.Add(text5, pos=(6, 0), flag=wx.TOP|wx.LEFT, border=10)

        self.tc6 = wx.TextCtrl(panel)
        sizer.Add(self.tc6, pos=(6, 1), span=(1, 3), flag=wx.TOP|wx.EXPAND,
            border=5)
        #combo = wx.ComboBox(panel)
        #sizer.Add(combo, pos=(4, 1), span=(1, 3),
        #    flag=wx.TOP|wx.EXPAND, border=5)

        button6 = wx.Button(panel, label="Browse...")
        button6.Bind(wx.EVT_BUTTON, self.onEventOpenFile)
        sizer.Add(button6, pos=(6, 4), flag=wx.TOP|wx.RIGHT, border=5)

        
        
        
        
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

    def onEventOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile="",
            wildcard=wildcard1,
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            #print(paths)
            #print("You chose the following file(s):")
            for path in paths:
                self.input_event_file = path
                self.tc6.SetValue(self.input_event_file)
                #print(path)
        #print(self.input_struct_file)
        dlg.Destroy()

        
    def OnOkApp(self, event):
        
        dict_file={}
        
        ip_address = self.tc1.GetValue()
        dict_file['scanner_ipaddress'] = ip_address
        
        socket_host = self.tc2.GetValue()
        dict_file['scanner_socket'] = socket_host
        
        s_fn = self.input_struct_file
        dict_file['struct_image'] = s_fn
        
        f_fn=self.input_func_file
        dict_file['func_image'] = f_fn

        e_fn=self.input_event_file
        dict_file['event'] = e_fn

        print(dir_path)
        if s_fn is None or ''==s_fn:
           wx.MessageBox('Please Upload Structural Image File', 'Warning', wx.OK | wx.ICON_WARNING)
           return
        if f_fn is None or ''==s_fn:
           wx.MessageBox('Please Upload Functional Image File', 'Warning', wx.OK | wx.ICON_WARNING)
           return
        if e_fn is None or ''==e_fn:
           wx.MessageBox('Please Upload Event File', 'Warning', wx.OK | wx.ICON_WARNING)
           return
       
        
        setting_file=dir_path+'/'+'setting.yaml'
        
        with open(setting_file, 'w') as file:
            documents = yaml.dump(dict_file, file)
        self.Close()
        #sdataset, fdataset = prepRealDataset(s_fn, f_fn)
        # run pynealScanner Simulator
        #print(ip_address)
        #print(socket_host)
        #print(s_fn)
        #print(f_fn)
        #scannerSimulator(sdataset, fdataset, TR=1000, host=ip_address,port=socket_host1)


    def OnQuitApp(self, event):
        
        self.Close()
        
    def onChecked(self, e): 
      cb = e.GetEventObject() 
      #print(cb.GetLabel(),' is clicked',cb.GetValue())

def launchScanner():
    """Main Scanner Loop.

    This function will launch setup GUI, retrieve settings, initialize all
    threads, and start processing incoming scans

    """
    app = wx.App()
    ex = Example(None, title="Scanner Simulation Module")
    ex.Show()
    app.MainLoop()
    
    
def isValidParameter(stringValue):
     
    if stringValue is not None and stringValue.strip()!='':
       return stringValue
    else:
       return None
    
    

if __name__ == '__main__':
    launchScanner()
    
    setting_file=dir_path+'/'+'setting.yaml'
    
    with open(setting_file) as file:
        
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format

        path = dir_path+'/graph/'
        rot_file_name1 = path+'rotational.txt'
        trans_file_name1 = path+'translation.txt'
        motion_file_name1 = path+'motion.txt'
        writtingFile( rot_file_name1 , '')
        writtingFile( trans_file_name1 , '')
        writtingFile( motion_file_name1 , '')
        
        parameter_list = yaml.load(file, Loader=yaml.FullLoader)
        
        if parameter_list is not None:
            
            ip_address = parameter_list['scanner_ipaddress'] 
            
            socket_host = parameter_list['scanner_socket'] 
            
            s_fn = parameter_list['struct_image']
            
            f_fn = parameter_list['func_image']

            e_fn = parameter_list['event']
            
            if isValidParameter(ip_address) != None and  isValidParameter(socket_host) != None and isValidParameter(s_fn) != None and isValidParameter(f_fn) != None and isValidParameter(e_fn) != None:
    
               sdataset, fdataset, eventsdataset = prepRealDataset(s_fn, f_fn, e_fn)

               writtingFile( dir_path+'/temp/'+'file_event3.tsv' , eventsdataset )
               
               scannerSimulator(sdataset, fdataset, eventsdataset, TR=1600, host=ip_address,port=socket_host)
               
               
                
               #app.MainLoop()
