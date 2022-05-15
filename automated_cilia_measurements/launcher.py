from imp import create_dynamic
import sys
import os
import argparse
from automated_cilia_measurements.pixels_to_measurement import (
    main as pixels_to_measurement,
)
from automated_cilia_measurements.center2center import main as c2c
from automated_cilia_measurements.clustering import main as clustering
from automated_cilia_measurements.label_cprof_im import main as label_cprof_im
from automated_cilia_measurements.data_table import main as data_table
from automated_cilia_measurements.label_c2c import main as label_c2c
from automated_cilia_measurements.label_valid_cilia import main as organelle_labeler
from automated_cilia_measurements.check_accuracy import main as check_accuracy
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ic",
        "--input_csvs",
        help="input csvs from cellprofiler",
        required=True,
    )

    parser.add_argument(
        "-ii",
        "--input_images",
        help="input images from cellprofiler",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="output folder",
        required=True,
    )

    parser.add_argument(
        "-f",
        "--factor",
        help="conversion factor. enter number if you want to convert pixels to micrometer",
        required=False,
    )

    parser.add_argument(
        "-hr",
        "--heirarchical",
        help="make dendrograms",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-x",
        "--xmeans",
        help="perform xmeans clustering",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-p",
        "--pca",
        help="perform dimensionality reduction",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-u", "--umap", help="perform UMAP", required=False, action="store_true"
    )

    parser.add_argument(
        "-ce",
        "--cellprofiler_labeling",
        help="label cellprofiler images",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-nce",
        "--num_cellprofiler_images",
        help="number of images to label for cellprofiler image labeling, if specific number of im wanted",
        required=False,
    )

    parser.add_argument(
        "-cce",
        "--centriole_cellprofiler_images",
        help="label centriole images for cellprofiler image labeling",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-dt",
        "--data_table",
        help="make a data table",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-cc",
        "--label_c2c",
        help="label c2c images",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-ncc",
        "--num_c2c_images",
        help="number of images to label for c2c image labeling, if specific number of im wanted",
        required=False,
    )

    parser.add_argument(
        "-nl",
        "--nuclei_label",
        help="label nuclei images",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-nnl",
        "--num_nuclei_images",
        help="number of images to label for nuclei image labeling, if specific number of im wanted",
        required=False,
    )

    parser.add_argument(
        "-cl",
        "--cilia_label",
        help="label cilia images",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-ncl",
        "--num_cilia_images",
        help="number of images to label for cilia image labeling, if specific number of im wanted",
        required=False,
    )

    parser.add_argument(
        "-cel",
        "--cent_label",
        help="label centriole images",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-ncel",
        "--num_cent_images",
        help="number of images to label for centriole image labeling, if specific number of im wanted",
        required=False,
    )

    parser.add_argument(
        "-ac",
        "--true_results_for_accuracy_checker",
        help="true results of cilia path, if accuracy checker wanted",
        required=False,
    )
    return vars(parser.parse_args())


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
        print("In delete_widget")

        label_name = f"Label {widget_name}"
        print(widget_name)
        print(label_name)

        print(self[widget_name])
        print(self[label_name])
        print(self.cur_widgets)
        self.inside.remove_widget(self[widget_name])
        self.cur_widgets.remove(widget_name)

        self.inside.remove_widget(self[label_name])
        self.cur_widgets.remove(label_name)

        print(self[widget_name])
        print(self[label_name])
        print(self.cur_widgets)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except:
            return "Doesn't exist!"

    def create_dynamic_checkbox_handler(self, parent_widget_name, dependencies):
        def on_checkbox_handler(checkbox, value):
            print("In handler")
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

    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)
        self.cols = 1

        self.inside = GridLayout() # Create a new grid layout
        self.inside.cols = 2 # set columns for the new grid layout

        self.cur_widgets = []

        self.append_widget(
            "Input CSVs (from CellProfiler):", "input_csvs", TextInput(multiline=False)
        )
        self.append_widget(
            "Input images (from CellProfiler):",
            "input_images",
            TextInput(multiline=False),
        )
        self.append_widget("Output folder:", "output", TextInput(multiline=False))

        parent_name = "microm"
        cb = CheckBox()
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

        self.append_widget("Make dendograms?", "heirachical", CheckBox())
        self.append_widget("Make XMeans?", "xmeans", CheckBox())
        self.append_widget("Perform PCA", "pca", CheckBox())
        self.append_widget("Perform UMAP", "umap", CheckBox())

        parent_name = "cellprofiler_labeling"
        cb = CheckBox()
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label for cellprofiler image labeling, if specific number of im wanted:",
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

        self.append_widget("Make a data table?", "data_table", CheckBox())

        parent_name = "label_c2c"
        cb = CheckBox()
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label for c2c image labeling, if specific number of im wanted:",
                        "num_c2c_images",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Visualize c2c output?", parent_name, cb)

        parent_name = "nuclei_label"
        cb = CheckBox()
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label for nuclei image labeling, if specific number of im wanted:",
                        "num_nuclei_images",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Visualize nuclei on images?", parent_name, cb)

        parent_name = "cilia_label"
        cb = CheckBox()
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label for cilia image labeling, if specific number of im wanted:",
                        "num_cilia_images",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Visualize valid cilia on images?", parent_name, cb)

        parent_name = "cent_label"
        cb = CheckBox()
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "Number of images to label for centriole image labeling, if specific number of im wanted:",
                        "num_cent_images",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Visualize valid centriole on images?", parent_name, cb)

        parent_name = "accuracy_check"
        cb = CheckBox()
        cb.bind(
            active=self.create_dynamic_checkbox_handler(
                parent_name,
                [
                    (
                        "True results of cilia path, if accuracy checker wanted:",
                        "true_results_for_accuracy_checker",
                        TextInput(multiline=False),
                    ),
                ],
            )
        )
        self.append_widget("Check accuracy?", parent_name, cb)

        # Primary content
        self.add_widget(self.inside)
        self.submit = Button(text="Submit")  
        self.add_widget(self.submit)


class Gui(App):
    def build(self):
        return MyGrid()


def gui():
    Gui().run()


def main():
    args = parse_args()
    dir_out = args["output"]

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    csvs_in = args["input_csvs"]
    images_in = args["input_images"]

    if args.get("factor"):
        microm_conversion_path = os.path.join(dir_out, "microm_converted")

        if not os.path.exists(microm_conversion_path):
            os.mkdir(microm_conversion_path)

        pixels_to_measurement(
            measurements=csvs_in, factor=args["factor"], output=microm_conversion_path
        )

        csvs_in = microm_conversion_path

    c2c_output_path = os.path.join(dir_out, "c2c_output")

    if not os.path.exists(c2c_output_path):
        os.mkdir(c2c_output_path)

    c2c(input=csvs_in, output=c2c_output_path)

    if (
        args.get("xmeans")
        or args.get("pca_features")
        or args.get("heirarchical")
        or args.get("umap")
    ):

        clustering(
            measurements=csvs_in,
            c2c=os.path.join(c2c_output_path, "c2coutput.csv"),
            xmeans=args.get("xmeans"),
            pca_features=args.get("pca_features"),
            heirarchical=args.get("heirarchical"),
            umap=args.get("umap"),
        )

    if args.get("cellprofiler_labeling"):
        cprof_vis_output = os.path.join(dir_out, "cprof_vis_output")

        if not os.path.exists(cprof_vis_output):
            os.mkdir(cprof_vis_output)

        label_cprof_im(
            input=csvs_in,
            images=images_in,
            output=cprof_vis_output,
            num=args.get("num_cellprofiler_images"),
            centriole=args.get("centriole_cellprofiler_images"),
        )

    if args.get("data_table"):
        data_tbl_output = os.path.join(dir_out, "data_tbl")

        if not os.path.exists(data_tbl_output):
            os.mkdir(data_tbl_output)

        data_table(input=csvs_in, c2c=c2c_output_path, output=data_tbl_output)

    if args.get("label_c2c"):
        c2c_vis_output = os.path.join(dir_out, "c2c_vis_output")

        if not os.path.exists(c2c_vis_output):
            os.mkdir(c2c_vis_output)

        label_c2c(
            input=csvs_in,
            images=images_in,
            c2c=c2c_output_path,
            num=args.get("num_c2c_images"),
            output=c2c_vis_output,
        )

    if args.get("nuclei_label"):
        channel_lbl = os.path.join(dir_out, "channel_NucleusOverlay_vis_output")
        if not os.path.exists(channel_lbl):
            os.mkdir(channel_lbl)

        organelle_labeler(
            measurements=csvs_in,
            output=channel_lbl,
            images=images_in,
            num=args.get("num_nuclei_images"),
            channel="01",
        )

    if args.get("cilia_label"):
        channel_lbl = os.path.join(dir_out, "channel_CiliaOverlay_vis_output")
        if not os.path.exists(channel_lbl):
            os.mkdir(channel_lbl)

        organelle_labeler(
            measurements=csvs_in,
            output=channel_lbl,
            images=images_in,
            num=args.get("num_cilia_images"),
            channel="02",
            c2c=c2c_output_path,
        )

    if args.get("cent_label"):
        channel_lbl = os.path.join(dir_out, "channel_CentrioleOverlay_vis_output")
        if not os.path.exists(channel_lbl):
            os.mkdir(channel_lbl)

        organelle_labeler(
            measurements=csvs_in,
            output=channel_lbl,
            images=images_in,
            num=args.get("num_cent_images"),
            channel="03",
            c2c=c2c_output_path,
        )

    if args.get("true_results_for_accuracy_checker"):
        accuracy_path = os.path.join(dir_out, "check_accuracy")
        if not os.path.exists(accuracy_path):
            os.mkdir(accuracy_path)

        check_accuracy(
            true=args["true_results_for_accuracy_checker"],
            output=accuracy_path,
            c2c=os.path.join(c2c_output_path, "new_cilia.csv"),
        )


if __name__ == "__main__":
    main()
