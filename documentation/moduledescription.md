# BLAST User Interface Module Description

Short description of each module and the main methods.


1. ```Gui.py``` 
top level tkinter event loop
&nbsp;

	2. ```BlastGUI.py```

        creates the main GUI structure. There are four main data structures in the ```__init__()```:

        * ```self.data``` a dict of studies for the currently loaded case. Studies have a number of 3d image volumes and overlay volumes as defined in ```DcmCase.py```.  
        * ```self.blastdata``` a dict of studies, containing the various parameters and values associated with BLAST processing.
        * ```self.rois``` are BLAST and SAM segmentation masks and are defined in ```ROI.py```
        * ```self.pt``` is a list of points clicked during BLAST point selection workflow, defined in ```ROI.py```  

        main methods:

        * ```runblast()``` - the main method for all BLAST workflows
        * a number of utility methods that are shortcuts for referencing various methods in between various modules  
        &nbsp;

		3. ```CreateFrame.py``` - a base class for different parts of UI
	            
            3.a. ```CreateSliceViewerFrame(CreateFrame)``` - a subclass of ```CreateFrame``` that provides a base class for different slice viewers
        
            main methods:
        
            * ```resizer()``` - for dragging window size
            * ```create_blank_canvas()``` initializes the ```FigureCanvasTkAgg``` graphics panel with a blank screen
            * a number of keyboard mouse and touchpad methods. 
        
                3.a.1. ```CreateSAMSVFrame(CreateSliceViewerFrame)``` - a subclass of ```CreateSliceViewerFrame``` for the SAM viewer

                main methods:
        
                * ```resize()``` - separate resizing calculation for root window when sliceviewer changes
                * ```create_canvas()``` - creates the ```FigureCanvasTkAgg``` graphics panel when dataset first loaded. 
                * ```update_slice()``` - called generically for just about any update to the display
                * a number of other lesser update methods for smaller things
                * ```normalslice_callback()``` - stats for the BLAST workflow
                * ```sam2d(),sam3d()``` - callbacks for running the SAM inference
                * several other keyboard/event/bbox methods relating to various workflow steps
                        
            3.b. ```CreateSAMROIFrame(CreateFrame)``` - a subclass of ```CreateFrame``` that provides the various ROI functions
        
            main methods:
            * ```layer_callback(), layerROI_callback(), layerSAM_callback()``` - for creating the fusion overlay of the raw BLAST mask, final BLAST ROI, and SAM ROI respectively.
            * a number of lesser update methods. 
            * ```enhancingROI_overlay(),finalROI_overlay(),SAM_overlay()``` - for on/off control of the raw BLAST mask, final BLAST ROI and SAM ROI overlays respectively
            * numerous methods specifically for handling the workflow of ROI selection: 

                * ```selectROI()``` - callback from the button 'selectROI'
                * ```createROI()``` - add a new empty ROI to the list
                * ```ROIclick()``` - processes the ROI selection button click event in the graphics window
                * ```closeROI()``` - uses 3d connected objects to finalize the selected 3d object after a click
                * ```updateROI()``` - update an existing ROI after changes
                * ```saveROI()``` - callback from the button 'saveROI'
                * ```clearROI()``` - callback from the button 'clearROI', erases the last/most recent ROI in the list
                * ```resetROI()``` - erase all ROI's
                * ```ROIstats()``` - optional during saveROI

            * methods for SAM prompts:

                * ```save_prompts()``` - create 2d slice prompts from 3d volume
                * ```put_prompts_remote(),get_predictions_remote()``` - upload/download from cloud
                * ```segment_sam()``` - run the SAM inference
                * ```load_sam()``` - load inference results back to viewer
            
            * methods for moving/copying data between structures as a consequence of various workflows:

                ```updateROIData(), updateBLAST(), updateData(), updateSAMData()``` 

            * about a dozen methods (not listed) for the BLAST point selection workflow. This has grown to become a large amount of code and indicates that this module is probably too big and should be split up. 
        
            3.c. ```CreateCaseFrame(CreateFrame)``` - routine file handling methods. 

            &nbsp;

        4. Other Modules

            4.a. ```DcmCase.py``` - includes several classes for studies and cases
            
            * ```Study.__init__()``` - the main data structure.  
            
            main dicom processing methods:
            
            * ```load_data()``` - load dicom images and read protocol/sequence from header 
            * ```get_affine()``` - read affine from dicom header for use in nifti headers
            * ```preprocess()``` - main method for the processing pipeline
            * ```normalstats()``` - duplicate of ```normalslice_callback()``` for creating z-score images
            * ```segment()``` - optional nnUNet
            * ```extract_brain()``` - optional skull removal 
            * ```n4_bias()``` - optional bias field correction
            * ```register()``` - optional registration to MNI coordinates
            
            4.b. ```Blastbratsv3.py``` - a port of the original matlab algorithm. 
            
            4.c. ```SAM``` package - largely huggingface code
            * ```SAM.main()``` - the former ```sam_hf.py``` standalone script.
            * ```SSHSession.py``` - only half-complete as yet, pending further work on the design of the web app. 
            
            4.d. ```OverlayPlots.py``` - the various plots are ad hoc and do not make use of any base classes or common processing conventions. 

            4.e. ```ROI.py``` - several classes for roi's. 

