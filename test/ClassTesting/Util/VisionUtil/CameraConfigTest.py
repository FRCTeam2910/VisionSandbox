import src.Util.VisionUtil.CameraConfig as CameraConfig

CameraConfig = CameraConfig.CameraConfig

cameraConfig = CameraConfig('test/MiscTestScripts/CameraConfig.cfg')

print(cameraConfig.__dict__)