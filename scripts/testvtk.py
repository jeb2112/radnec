import vtk


class DICOMViewer:
    def __init__(self, folder_path):
        # Read DICOM series
        self.reader = vtk.vtkDICOMImageReader()
        self.reader.SetDirectoryName(folder_path)
        self.reader.Update()

        # Setup the renderer, render window, and interactor
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # Setup the image actor and slice mapper
        self.image_slice_mapper = vtk.vtkImageResliceMapper()
        self.image_slice_mapper.SetInputConnection(self.reader.GetOutputPort())
        # self.image_slice_mapper.SetSliceNumber(0)
        self.image_slice_mapper.SliceFacesCameraOn()
        self.image_slice_mapper.SliceAtFocalPointOn()

        self.image_actor = vtk.vtkImageSlice()
        self.image_actor.SetMapper(self.image_slice_mapper)

        self.renderer.AddViewProp(self.image_actor)

        # Set up camera for axial view
        self.renderer.GetActiveCamera().ParallelProjectionOn()

        # Add interactivity
        self.interactor_style = SliceScrollInteractorStyle(self)
        self.interactor.SetInteractorStyle(self.interactor_style)

    def start(self):
        # Adjust the window size and render
        self.render_window.SetSize(800, 800)
        self.renderer.ResetCamera()
        self.render_window.Render()
        self.interactor.Start()


class SliceScrollInteractorStyle(vtk.vtkInteractorStyleImage):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.slice_min = 0
        self.slice_max = viewer.reader.GetOutput().GetExtent()[5]
        self.current_slice = 0

    def OnMouseWheelForward(self):
        # Scroll forward through slices
        self.current_slice = min(self.current_slice + 1, self.slice_max)
        self.update_slice()

    def OnMouseWheelBackward(self):
        # Scroll backward through slices
        self.current_slice = max(self.current_slice - 1, self.slice_min)
        self.update_slice()

    def update_slice(self):
        # Update the displayed slice
        self.viewer.image_slice_mapper.SetSliceNumber(self.current_slice)
        self.viewer.render_window.Render()


if __name__ == "__main__":
    import argparse

    # Parse folder path from command line
    parser = argparse.ArgumentParser(description="DICOM Viewer")
    parser.add_argument("--folder", type=str, help="Path to folder containing DICOM series")
    args = parser.parse_args()

    # Run the viewer
    viewer = DICOMViewer(args.folder)
    viewer.start()