import os
from kivy.app import App
from kivy.properties import BooleanProperty
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from automated_cilia_measurements.launcher import main as run_scripts

# Global instance of the application, used to stop the gui programmatically
instance = None


class MyGrid(GridLayout):
    def append_widget(self, label_text, widget_name, widget, index=None):
        self[widget_name] = widget
        label_name = f"Label {widget_name}"
        self[label_name] = Label(text=label_text)

        if index is None:
            self.inside.add_widget(self[label_name])
            self.cur_widgets.append(label_name)

            self.inside.add_widget(self[widget_name])
            self.cur_widgets.append(widget_name)
        else:
            # Invert because grid layout indexes the bottom of the screen as 0
            self.inside.add_widget(self[widget_name], index=-index)
            self.cur_widgets.insert(index, widget_name)

            self.inside.add_widget(self[label_name], index=-index)
            self.cur_widgets.insert(index, f"Label {widget_name}")

    def delete_widget(self, widget_name):
        label_name = f"Label {widget_name}"
        self.inside.remove_widget(self[widget_name])
        self.cur_widgets.remove(widget_name)

        self.inside.remove_widget(self[label_name])
        self.cur_widgets.remove(label_name)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except:
            None

    def create_dynamic_checkbox_handler(self, parent_widget_name, dependencies):
        def on_checkbox_handler(_, value):
            parent_index = self.cur_widgets.index(parent_widget_name)

            if value:
                for (dep_label, dep_widget_name, widget) in reversed(dependencies):
                    self.append_widget(
                        dep_label, dep_widget_name, widget, (parent_index + 1)
                    )
            else:
                for (_, dep_widget_name, _) in dependencies:
                    self.delete_widget(dep_widget_name)

        return on_checkbox_handler

    def submit_callback(self, _):
        instance.stop()
        run_scripts(
            output=self["output"],
            input_csvs=self["input_csvs"],
            input_images=self["input_images"],
            factor=self["factor"] and self["factor"].text,
            xmeans=self["xmeans"].active,
            pca_features=self["pca_features"].active,
            heirarchical=self["heirarchical"].active,
            umap=self["umap"].active,
            measurements_per_im=self['measurements_per_im'].active, 
            scatters=self['scatters'].active,
            hist=self['hist'].active,
            bins=self['bins'].active,
            cellprofiler_labeling=self["cellprofiler_labeling"].active,
            num_cellprofiler_images=self["num_cellprofiler_images"]
            and self["num_cellprofiler_images"].text,
            centriole_cellprofiler_images=self["centriole_cellprofiler_images"]
            and self["centriole_cellprofiler_images"].active,
            data_table=self["data_table"].active,
            label_c2c=self["label_c2c"].active,
            num_c2c_images=self["num_c2c_images"] and self["num_c2c_images"].text,
            nuclei_label=self["nuclei_label"].active,
            num_nuclei_images=self["num_nuclei_images"]
            and self["num_nuclei_images"].text,
            cilia_label=self["cilia_label"].active,
            num_cilia_images=self["num_cilia_images"] and self["num_cilia_images"].text,
            cent_label=self["cent_label"].active,
            num_cent_images=self["num_cent_images"] and self["num_cent_images"].text,
            true_results_for_accuracy_checker=self["true_results_for_accuracy_checker"]
            and self["true_results_for_accuracy_checker"],
        )

    def file_chooser_widget(self, widget_name):
        box = BoxLayout()

        file_chooser = FileChooserListView(path=os.getcwd(), dirselect=True)

        box.add_widget(file_chooser)

        popup = Popup(title="Double click to choose desired file", content=box)
        if not hasattr(self, "popup_refs"):
            self.popup_refs = []
        self.popup_refs.append(popup)
        file_chooser_button = Button(text="Choose file", size_hint_y=None, height=50)
        file_chooser_button.bind(on_press=popup.open)

        def file_chooser_handler(*args):
            self[widget_name] = file_chooser.selection[0]
            for popup in self.popup_refs:
                popup.dismiss()

        file_chooser.bind(selection=file_chooser_handler)
        return file_chooser_button

    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)
        self.cols = 1

        self.inside = GridLayout()  # Create a new grid layout
        self.inside.cols = 2  # set columns for the new grid layout

        self.cur_widgets = []

        self.append_widget(
            "Input CSVs (from CellProfiler):",
            "input_csvs",
            self.file_chooser_widget("input_csvs"),
        )

        self.append_widget(
            "Input images (from CellProfiler):",
            "input_images",
            self.file_chooser_widget("input_images"),
        )
        self.append_widget(
            "Output folder:", "output", self.file_chooser_widget("output")
        )

        parent_name = "microm"
        color_check = [255, 255, 255, 2]
        cb = CheckBox(color=color_check)
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Scale factor to convert pixels to micrometers:",
                        "factor",
                        TextInput(multiline=False),
                    )
                ],
            )
        )
        self.append_widget("Convert pixels to micrometers?", parent_name, cb)

        self.append_widget(
            "Make dendograms?", "heirarchical", CheckBox(color=color_check)
        )
        self.append_widget("Make XMeans?", "xmeans", CheckBox(color=color_check))
        self.append_widget("Perform PCA", "pca_features", CheckBox(color=color_check))
        self.append_widget("Perform UMAP", "umap", CheckBox(color=color_check))

        self.append_widget(
            "Make measurements per image bars?", "measurements_per_im", CheckBox(color=color_check)
        )
        self.append_widget("Make scatterplots?", "scatters", CheckBox(color=color_check))

        parent_name = "hist"
        cb = CheckBox(color=color_check)
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Do you want to enter in custom bins for the histograms?",
                        "bins",
                        CheckBox(color=color_check),
                    ),
                ],
            )
        )
        self.append_widget("Make histograms ?", parent_name, cb)

        parent_name = "cellprofiler_labeling"
        cb = CheckBox(color=color_check)
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label, if specific:",
                        "num_cellprofiler_images",
                        TextInput(multiline=False),
                    ),
                    (
                        "Label centriole images for cellprofiler image labeling?",
                        "centriole_cellprofiler_images",
                        CheckBox(),
                    ),
                ],
            )
        )
        self.append_widget("Perform labeling of CellProfiler images?", parent_name, cb)

        self.append_widget(
            "Make a data table?", "data_table", CheckBox(color=color_check)
        )

        parent_name = "label_c2c"
        cb = CheckBox(color=color_check)
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label, if specific:",
                        "num_c2c_images",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Visualize c2c output?", parent_name, cb)

        parent_name = "nuclei_label"
        cb = CheckBox(color=color_check)
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label, if specific:",
                        "num_nuclei_images",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Visualize nuclei on images?", parent_name, cb)

        parent_name = "cilia_label"
        cb = CheckBox(color=color_check)
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label, if specific:",
                        "num_cilia_images",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Visualize valid cilia on images?", parent_name, cb)

        parent_name = "cent_label"
        cb = CheckBox(color=color_check)
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label, if specific:",
                        "num_cent_images",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Visualize valid centriole on images?", parent_name, cb)

        parent_name = "accuracy_check"
        cb = CheckBox(color=color_check)
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "True results of cilia path, if accuracy checker wanted:",
                        "true_results_for_accuracy_checker",
                        self.file_chooser_widget("true_results_for_accuracy_checker"),
                    ),
                ],
            )
        )
        self.append_widget("Check accuracy?", parent_name, cb)

        # Primary content
        self.add_widget(self.inside)
        self.submit = Button(text="Submit", size_hint_y=None, height=50)
        self.submit.bind(on_press=self.submit_callback)
        self.add_widget(self.submit)


class Gui(App):
    def build(self):
        return MyGrid()


def main():
    global instance
    instance = Gui()
    instance.run()


if __name__ == "__main__":
    main()

