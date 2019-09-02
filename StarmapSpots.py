#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## import statements
import functools
import os
from typing import *
import click
import numpy as np
from skimage.io import imread
from slicedimage import ImageFormat
from slicedimage.backends import CachingBackend, DiskBackend, HttpBackend, SIZE_LIMIT
from six.moves import urllib
import tempfile
import pickle
import pandas as pd
import shutil

import starfish
from starfish import *
from starfish.experiment.builder import FetchedTile, TileFetcher, write_experiment_json
from starfish.types import Axes, Coordinates, CoordinateValue, Features
from starfish.core.types import DecodedSpots
from starfish.image import Filter, LearnTransform, ApplyTransform
from starfish.core.image._registration.transforms_list import TransformsList
from starfish.util.plot import (
    diagnose_registration, imshow_plane, intensity_histogram
)


##Function to parse filename to get the index of the gene from the imported gene table
def parseFilename(filename: str) -> Tuple[int, int, int]:
    filename = os.path.splitext(os.path.basename(filename))[0]
    tokens: List[str] = filename.split('_')
    sec: int = int(tokens[0][1:])
    r  : int = int(tokens[1][1:])
    ch : int = int(tokens[2][1:])
    return (sec, r, ch)

## classes to format image tiles
class StarMapTile(FetchedTile):

    def __init__(
            self,
            file_path: str,
            coordinates: Mapping[Union[str, Coordinates], CoordinateValue]
    ) -> None:
        self.file_path = file_path
        self._coordinates = coordinates

    @property
    def shape(self) -> Mapping[Axes, int]:
        shape = np.shape(self.tile_data())
        return {Axes.X: shape[1], Axes.Y: shape[0]}

    @property
    def coordinates(self) -> Mapping[Union[str, Coordinates], CoordinateValue]:
        return self._coordinates
    
    def tile_data(self) -> np.ndarray:
        return imread(self.file_path)
    
    def __str__(self) -> str:
        return self.file_path
    
class StarMapTileFetcher(TileFetcher):
    
    #because this is only ever reading in one file per time the program is run 
    #(instead of having to read in multiple rounds/channels like it normally would),
    #`input_dir` is really a filename, but can't change the parameter name due to subclassing.
    def __init__(self, input_dir: str) -> None:
        self.input_dir = input_dir
        
    def get_tile( #most of these parameters don't matter because we're reading one file
            self, fov_id: int, round_label: int, ch_label: int, zplane_label: int) -> FetchedTile:
        coordinates = {
            Coordinates.X: (0.0, 1.0),
            Coordinates.Y: (0.0, 1.0),
            Coordinates.Z: (0.0, 1.0),
        }
        return StarMapTile(self.input_dir, coordinates)
    
## Function to format the data and output a complete experiment with correct json scheme
def format_data(input_dir: str, output_dir: str, gene_name: str) -> None:
    
    primary_image_dimensions: Mapping[Axes, int] = {
        Axes.ROUND: 1,
        Axes.CH: 1,
        Axes.ZPLANE: 1
    }
        
    aux_name_to_dimensions: Mapping[str, Mapping[Union[str, Axes], int]] = {
        "nissl": {
            Axes.ROUND: 1,
            Axes.CH: 1,
            Axes.ZPLANE: 1
        }
    }
    
    write_experiment_json(
        path=output_dir,
        fov_count=1,
        tile_format=ImageFormat.TIFF,
        primary_image_dimensions=primary_image_dimensions,
        aux_name_to_dimensions=aux_name_to_dimensions,
        primary_tile_fetcher=StarMapTileFetcher(input_dir),
        aux_tile_fetcher={"nissl": StarMapTileFetcher(input_dir)},
        dimension_order=(Axes.ROUND, Axes.CH, Axes.ZPLANE)
    )
    
    codebook = [
        {
            Features.CODEWORD: [
                {Axes.ROUND.value: 0, Axes.CH.value: 0, Features.CODE_VALUE: 1}
            ],
            Features.TARGET: gene_name
        }
    ]
        
        
    Codebook.from_code_array(codebook).to_json(os.path.join(output_dir, "codebook.json"))

##Function to pre-process an `ImageStack`
def preprocess_fov(primary_fov_imagestack: ImageStack,
                  n_processes: Optional[int] = None,
                  is_volume: Optional[bool] = False,
                  verbose: Optional[bool] = False) -> ImageStack:
    """Preprocess a Starfish field of view image stack in preparation for
    spot/pixel finding.

    NOTE: This preprocessing pipeline processes imagestacks in place!

    Args:
       primary_fov_imagestack (ImageStack): A starfish FOV Imagestack
       n_processes (Optional[int]): Number of processes to use for
           preprocessing steps. If None, uses the output of os.cpu_count().
           Defaults to None.

    Returns:
       ImageStack: A preprocessed starfish imagestack.
    """
    print("Applying First Clip...")
    first_clip = Filter.ClipPercentileToZero(p_min=75, p_max=100,
                                            is_volume=is_volume)
    first_clip.run(primary_fov_imagestack, in_place=True, verbose=verbose,
                  n_processes=n_processes)

    print("Applying Bandpass...")
    bpass = Filter.Bandpass(lshort=0.5, llong=7, threshold=1/(1<<16-1),
                           is_volume=is_volume)
    bpass.run(primary_fov_imagestack, in_place=True, verbose=verbose,
             n_processes=n_processes)

    print("Applying Second Clip...")
    second_clip = Filter.ClipValueToZero(v_min=1/(1<<16-1), is_volume=is_volume)
    second_clip.run(primary_fov_imagestack, in_place=True, verbose=verbose,
                   n_processes=n_processes)

    print("Applying Gaussian Low Pass...")
    z_gauss_filter = Filter.GaussianLowPass(sigma=(1, 0, 0), is_volume=True)
    z_gauss_filter.run(primary_fov_imagestack, in_place=True,
                      n_processes=n_processes)

    print("Applying Final Clips...")
    final_percent_clip = Filter.ClipPercentileToZero(p_min=90, min_coeff=1.75)
    final_percent_clip.run(primary_fov_imagestack, in_place=True, verbose=verbose,
                          n_processes=n_processes)

    final_value_clip = Filter.ClipValueToZero(v_max=1000/(1<<16-1))
    final_value_clip.run(primary_fov_imagestack, in_place=True, verbose=verbose,
                        n_processes=n_processes)

    return primary_fov_imagestack

##Function to find and decode spots
def findAndDecodeSpots(fov: ImageStack, codebook: Codebook, verbose: Optional[bool] = False) -> IntensityTable:
    import starfish
    lmpf = starfish.spots.DetectSpots.LocalMaxPeakFinder(
        min_distance=2,
        stringency=0,
        min_obj_area=4,
        max_obj_area=600,
        verbose=verbose,
        is_volume=False
    )
    intensities: IntensityTable = lmpf.run(fov)
    return codebook.decode_per_round_max(intensities.fillna(0))

##Function to read in the gene table and return a linear list of gene names 
def geneTableToList(filename: str) -> List[str]:
    with open(filename, mode='r') as file:
        line: str = file.readline()
        return line.split(',')

##Function to import, filter, find spots, and output the `IntensityTable`
@click.command()
@click.option('--input', '-i', 'file_path', 
              required=True, 
              type=click.Path(exists=True), 
              help='Path to a TIFF image of a single section, round, channel slice.')
@click.option('--output', '-o', 
              envvar='PWD',
              type=click.Path(exists=True),
              help='Path to a folder to export the corresponding intensity table to.')
@click.option('--gene_table', '-g', 
              type=click.Path(exists=True),
              help='Path to the CSV containing a list of genes in order r0c0, r0c1, ..., r1c0, ...'
             )
@click.option('--num_channels', '-c',
              type=int,
              default=4,
              show_default=True,
              help='Number of channels (not including DAPI) in experiment. Only used for file and gene naming scheme purposes.'
             )
@click.option('--verbose', '-v', 
              count=True,
              help='When called one will output status of spot finder. When called twice will also output status of filters.'
             )

def cli(file_path: str, 
        output: str, 
        gene_table: str, 
        num_channels: int, 
        verbose: int
       ) -> None:
    
    click.echo(verbose)
    #Get the round and channel of the current file
    s, r, c = parseFilename(file_path) 
    gene_index: int = r * num_channels + c
    if gene_table:
        gene = geneTableToList(gene_table)[gene_index]
    else:
        gene = f"Gene {gene_index}"
    
    #creating a temporary directory to export the experiment to
    temp_dir = tempfile.TemporaryDirectory()
    exp_dir = temp_dir.name
    
    #writing the experiment
    format_data(
        input_dir=file_path,
        output_dir=exp_dir,
        gene_name=gene
    )

    #read in the experiment
    experiment: Experiment = Experiment.from_json(os.path.join(exp_dir, 'experiment.json'))
    fov: ImageStack = experiment["fov_000"].get_image("primary")

    #filter the fov,and find and decode spots
    preprocess_fov(fov, verbose=verbose>0)
    decoded = findAndDecodeSpots(fov, experiment.codebook, verbose=verbose>1)
    df = pd.DataFrame(dict(decoded['features'].coords))
    pixel_coordinates = pd.Index(['x', 'y', 'z'])
    ds = DecodedSpots(df)

    #save the output
    pickle.dump(ds.data, open(os.path.join(output,f'S{s}_R{r}_C{c}.spots'), mode='wb'))
    df.to_csv(path_or_buf=os.path.join(output,f'S{s}_R{r}_C{c}.csv'))

