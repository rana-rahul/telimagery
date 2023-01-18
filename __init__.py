#!/usr/bin/env python3
# coding: utf-8


from __future__ import annotations

import datetime
import os
import pathlib
import re
from typing import Optional, Tuple

import attr
import pandas as pd
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from skimage import filters
from strenum import StrEnum


class IMCCmap(StrEnum):
    """String enumeration for applying cmaps"""

    grey = "grey"
    red = "red"
    blue = "blue"
    green = "green"
    yellow = "yellow"
    cyan = "cyan"
    magenta = "magenta"


@attr.define
class IMCMassChannelMetadata:
    """A class that encapsulates metadata specifications for a given `IMCMassChannel`

    The properties of a mass channel/parameter are critical for downstream
    phenotypic analysis. These include: element, cell feature/biomarker and
    the atomic mass of the mass tag reporter(s). These can be extracted
    directly from the IMC .txt files since we don't have access to the
    database schema(s) for any binary files (for now).

    Attributes:
        element: the element coupled to the mass tag reporter
        mass: mass number associated with the given mass channel, a singular isotope
        cell_feature: the biomarker/bioactivity of interest; e.g. epitope or antigen
            for heavy-metal affinity probes, nucleic acid (DNA) for intercalators,
            tellurium for specific protein activity or cell state, or None
            for background mass channels such as Xe
    """

    element: str
    mass: int
    cell_feature: Optional[str]

    @staticmethod
    def from_txt_file(txt_file: pathlib.Path) -> IMCMassChannelMetadata:
        """Extracts mass channel metadata from an input .txt file

        This is slightly hacky since metadata extraction
        relies on the raw .txt files having stable, schematized
        names. Pattern matching can be improved here.
        Example .txt file: "174Yb-HLA-DR(Yb174Di).txt"

        Args:
            txt_file: incoming IMC .txt file to be parsed
        Returns:
            An IMCMassChannelMetadata object
        """

        # Check we have the right file name and type
        txt_file_name = os.path.basename(txt_file)
        assert txt_file_name.endswith(
            ".txt"
        ), f"Expected a .txt file, but got {txt_file_name}"

        # Strip terminal ".txt"
        dirty_channel_name = txt_file_name.rstrip(".txt")

        # Parse an expected file label for the mass numnber (Da or m/z),
        # element and target cell feature/marker
        if mass_tag := re.search(r"\((.*?)\)", dirty_channel_name):
            target_name = dirty_channel_name.strip(mass_tag.group(0)).lstrip("-")

            # TODO: for mypy, but should be cleaned up
            mass = re.search(r"(\d+)", mass_tag.group(1))  # type: ignore
            mass_number = mass.group(0)  # type: ignore
            element = mass_tag.group(1).split(mass_number)[0]

        # If there are mislabelled files, or a background channel is being
        # passed, then attempt alternative parsing
        else:
            print(
                f"Found unconventional .txt file label {txt_file_name}, trying again..."
            )

            # This is default for now, since the biomarker/epitope
            # names are heterogenous
            target_name = None

            # Extract all mass numbers, assert there is a singular value
            fetched_mass_numbers = re.findall(r"\d+", dirty_channel_name)
            assert len(set(fetched_mass_numbers)) == 1
            mass_number = fetched_mass_numbers[0]

            # Hacky way of parsing the element since it is almost always
            # appended to the front of the label
            element = dirty_channel_name.lstrip(str(mass_number))[:2]

        return IMCMassChannelMetadata(
            element=element, mass=int(mass_number), cell_feature=target_name
        )


@attr.define
class IMCMassChannel:
    """A data class capturing a particular IMC readout channel

    The intention here is to capture all the signal data from a
    single dimension or readout channel within a given IMC ablation.
    This is analogous to wavelength emission (in nm) for fluorescence
    techniques where we are interested in the fluorophore, emission wavelength,
    target cellular feature and a 2D heatmap.

    TODO:
        - add support for previewing images
        - add feature for auto-saving arrays to parent directory

    Attributes:
        metadata: mass channel properties as `IMCMassChannelMetadata`
        raw_image: the 'raw' 2D numpy array acquired post-ablation without modifcations
        height: height of region of interest (µm)
        width: hieght of ablated area; units in µm
        incomplete_scan: was the scan aborted prior to complete rasterization

    """

    metadata: IMCMassChannelMetadata
    raw_image: npt.NDArray
    height: int
    width: int
    incomplete_scan: bool

    @classmethod
    def from_txt_file(
        cls,
        txt_file: pathlib.Path,
    ) -> IMCMassChannel:
        """Parse a .txt file from IMC software into an `IMCAblation` object for analysis

        Note, we can also have incomplete ablations -- i.e. images where
        the ablation was stopped halfway across a row. This can be due to
        a number of factors, but we should be able to catch it and render
        the partial array.

        Args:
            txt_file: path to standard .txt file generated by IMC instrument for analysis
        Returns:
            A parsed, 2D heatmap for a singular cellular feature as `IMCAblation`;
                optionally provides an image preview
        """

        # Load the metadata
        metadata = IMCMassChannelMetadata.from_txt_file(txt_file)

        # Initially read the .txt file and load into pd.DataFrame
        raw_df = pd.read_csv(txt_file, sep="\t")

        # Extract signal intensities, labels and dimensions from table
        raw_array = raw_df.values

        # Adding to X and Y pixel counts to account for zero-indexing;
        x_size = raw_df["X"].max() + 1
        y_size = raw_df["Y"].max() + 1
        z_size = raw_df.shape[1]

        # Check for incomplete rasterization shape by verifying
        # number of pixels equals image dimension (area).
        if (x_size * y_size) == raw_array.shape[0]:

            # Determine the final height of an incomplete scan and
            # trim the image accordingly
            y_size -= 1
            delta = raw_array.shape[0] - (x_size * y_size)
            clipped = raw_array[:-delta]
            cleaned_image = np.reshape(clipped, [y_size, x_size, z_size], order="C")
            incomplete_scan = True
        else:
            cleaned_image = np.reshape(raw_array, [y_size, x_size, z_size], order="C")
            incomplete_scan = False

        return IMCMassChannel(
            metadata=metadata,
            raw_image=cleaned_image,
            height=y_size,
            width=x_size,
            incomplete_scan=incomplete_scan,
        )

    def plot_raw_array_histogram(self) -> None:
        """Plot a histogram of pixel intensities"""
        plt.hist(self.raw_image, bins=256)
        plt.show()

    def get_pixel_count(self) -> int:
        """Get total number of pixels in image; units are µ^2"""
        return self.height * self.width


@attr.define
class IMCAblation:
    """A class that encapsulates an image data for an entire IMC ablation

    An IMC ablation refers to all the image data acquired from scanning
    a particular region of interest on a stained tissue/cell sample. Here
    we capture all the constituent mass channels (see `IMCMassChannel`) and
    ablation metadata such as dimesions, date of acquisition, etc.


    Attributes:
        date: date of ablation; critical for bookkeeping experimental data
        mass_channels: collection of imaging data as collection of IMCMassChannel
    """

    date: datetime.date
    mass_channels: Tuple[IMCMassChannel, ...]


@attr.define
class ProcessedIMCImage(IMCMassChannel):
    """A class to model a processed image, its precursor and processing specifications

    Image transformation steps such as signal normalization, clamping,
    blurring and filtering are captured, along with the input array.

    Attributes:
        processing_config: parameters that describe image processing steps
        processed_image: resultant numpy array post modifications from `processing_config`
    """

    processing_config: ImageProcessingConfig
    processed_image: npt.NDArray


@attr.define
class ImageProcessingConfig:
    """Encapsualtes image processing steps

    TODO: Generalize this class to accommodate additional processing
    methods in addition to Gaussian filtering

    Attributes:
        normalized: are pixel intensities normalized between [0,1]
        clip_percentile: coerce maximnum pixel intensities to maximum threshold
        gaussian_sigma: standard deviation to use for Gaussian filtering
    """

    normalized: bool = False
    clip_percentile: Optional[float] = None
    gaussian_sigma: Optional[int] = None


def clip_img(img: npt.NDArray, clamp_percentile: float = 99.0) -> npt.NDArray:
    """Clips or coerces values in image array according to percentile

    This is desirable in image processing in general where
    artefact, (high intensity) pixels can skew the distribution of the signal.

    Args:
        img: the numpy array to be clamped
        clip_percentile: the threshold to be used in clipping
    Returns:
        A clamped image as numpy array
    """

    arr = np.copy(img)
    arr[arr > np.percentile(img, clamp_percentile)] = np.percentile(
        img, clamp_percentile
    )
    return arr


def gaussian_filter_img(img: npt.NDArray, sigma: int = 1) -> npt.NDArray:
    """Applies Gaussian kernel to input image

    Simply taken from:
    https://scikit-image.org/docs/dev/api/skimage.filters.html#gaussian

    Args:
        img: the numpy array to be filtered
        sigma: the standard deviation to use for the Gaussian kernel
    Returns:
        Filtered image as numpy array
    """

    arr = np.copy(img)
    output_arr = filters.gaussian(arr, sigma=sigma)

    return output_arr


def normalize_img(img: npt.NDArray) -> npt.NDArray:
    """Normalizes pixel intensities to [0,1]

    Args:
        img: the numpy array to be normalized
    Returns:
        Normalized image as numpy array
    """

    arr = np.copy(img)
    output_arr = arr / arr.max()
    return output_arr


def colour_image(img: npt.NDArray, cmap: IMCCmap) -> npt.NDArray:
    """Changes hue for a grey 2D heatmap to desired colour

    Args:
        img: the numpy array to be manipulated
    Returns:
        Normalized image as numpy array
    """

    input_array = np.copy(img)

    # We need an empty array for RGB and CYM stacking
    empty_img = np.zeros_like(input_array)

    red_arr = np.dstack((input_array, empty_img, empty_img))
    green_arr = np.dstack((input_array, img, empty_img))
    blue_arr = np.dstack((input_array, empty_img, img))

    if cmap == IMCCmap.grey:
        array = input_array
    elif cmap == IMCCmap.red:
        array = red_arr
    elif cmap == IMCCmap.green:
        array = green_arr
    elif cmap == IMCCmap.blue:
        array = blue_arr
    elif cmap == IMCCmap.cyan:
        array = green_arr + blue_arr
    elif cmap == IMCCmap.yellow:
        array = red_arr + green_arr
    elif cmap == IMCCmap.magenta:
        array = red_arr + blue_arr
    else:
        raise ValueError(
            f"Please provide a cmap option from: {[c.value for c in IMCCmap]}"
        )

    return array
