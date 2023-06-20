import PySpin
import cv2

class SingleShotCamera:
    def __init__(self):
        """
        Constructor
        """
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Get the first available camera
        self.cam = self.system.GetCameras()[0]

        # Initialize camera
        self.cam.Init()

        # configure camera
        # self.cam.UserSetSelector.SetValue(PySpin.UserSetDefault_Default)
        # self.cam.UserSetLoad()
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(10)
        self.cam.ExposureTime.SetValue(500)
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        
        # Begin acquiring images
        self.cam.BeginAcquisition()

    def capture_image(self):
        """
        Capture an image from the camera
        """
        try:
            

            # excute software trigger
            self.cam.TriggerSoftware.Execute()

            # Retrieve next received image and ensure image completion
            image = self.cam.GetNextImage()

            if image.IsIncomplete():
                print('Image incomplete with image status %d ...' % image.GetImageStatus())
            else:
                # Convert the raw image to an OpenCV image (BGR)
                img = cv2.cvtColor(image.GetNDArray(), cv2.COLOR_BayerBG2BGR_EA)

            return img

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return None

    def __del__(self):
        """
        Destructor
        """

        # End acquisition
        self.cam.EndAcquisition()

        # Deinitialize camera
        self.cam.DeInit()

        # Release reference to camera
        del self.cam

        # Clear camera list before releasing system
        self.system.ClearCameras()

        # Release system instance
        self.system.ReleaseInstance()
