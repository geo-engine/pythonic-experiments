use chrono::NaiveDate;
use futures::stream::BoxStream;
use futures::StreamExt;
use geoengine_datatypes::{
    primitives::TimeInterval,
    raster::{Grid2D, Pixel, Raster, RasterTile2D},
};
use geoengine_operators::engine::{
    ExecutionContext, InitializedOperator, InitializedOperatorBase, InitializedRasterOperator,
    InitializedVectorOperator, QueryContext, QueryProcessor, QueryRectangle, RasterOperator,
    RasterQueryProcessor, RasterResultDescriptor, TypedRasterQueryProcessor, VectorOperator,
};
use geoengine_operators::error::Error as GeoengineOperatorsError;
use geoengine_operators::util::Result;
use serde::{Deserialize, Serialize};

use ndarray::Array2;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::{types::PyModule, Py, Python};

/// An example operator that runs a pre post comparison on tiles using pca and kmeans
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KmeansOperator {
    pub raster_sources: Vec<Box<dyn RasterOperator>>,
    pub vector_sources: Vec<Box<dyn VectorOperator>>,
}

#[typetag::serde]
impl RasterOperator for KmeansOperator {
    fn initialize(
        mut self: Box<Self>,
        context: &dyn ExecutionContext,
    ) -> Result<Box<InitializedRasterOperator>> {
        if !self.vector_sources.is_empty() {
            return Err(GeoengineOperatorsError::InvalidNumberOfVectorInputs {
                expected: 0..1,
                found: self.vector_sources.len(),
            });
        }

        if self.raster_sources.len() != 1 {
            return Err(GeoengineOperatorsError::InvalidNumberOfRasterInputs {
                expected: 1..2,
                found: self.raster_sources.len(),
            });
        }

        let initialized_raster = self
            .raster_sources
            .pop()
            .expect("checked")
            .initialize(context)?;
        let result_descriptor = initialized_raster.result_descriptor().clone();

        let initialized_operator = InitializedPyOperator {
            raster_sources: vec![initialized_raster],
            vector_sources: vec![],
            result_descriptor,
            state: (),
        };

        Ok(initialized_operator.boxed())
    }
}

pub struct InitializedPyOperator {
    pub raster_sources: Vec<Box<InitializedRasterOperator>>,
    pub vector_sources: Vec<Box<InitializedVectorOperator>>,
    pub result_descriptor: RasterResultDescriptor,
    pub state: (),
}

impl InitializedOperatorBase for InitializedPyOperator {
    type Descriptor = RasterResultDescriptor;

    fn result_descriptor(&self) -> &Self::Descriptor {
        &self.result_descriptor
    }

    fn raster_sources(&self) -> &[Box<InitializedRasterOperator>] {
        &self.raster_sources
    }

    fn vector_sources(&self) -> &[Box<InitializedVectorOperator>] {
        &self.vector_sources
    }

    fn raster_sources_mut(&mut self) -> &mut [Box<InitializedRasterOperator>] {
        &mut self.raster_sources
    }

    fn vector_sources_mut(&mut self) -> &mut [Box<InitializedVectorOperator>] {
        &mut self.vector_sources
    }
}

impl InitializedOperator<RasterResultDescriptor, TypedRasterQueryProcessor>
    for InitializedPyOperator
{
    fn query_processor(&self) -> Result<TypedRasterQueryProcessor> {
        let typed_raster_processor = self.raster_sources[0].query_processor()?;

        Ok(match typed_raster_processor {
            TypedRasterQueryProcessor::U8(p) => {
                TypedRasterQueryProcessor::U8(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::U16(p) => {
                TypedRasterQueryProcessor::U16(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::U32(p) => {
                TypedRasterQueryProcessor::U32(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::U64(p) => {
                TypedRasterQueryProcessor::U64(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::I8(p) => {
                TypedRasterQueryProcessor::I8(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::I16(p) => {
                TypedRasterQueryProcessor::I16(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::I32(p) => {
                TypedRasterQueryProcessor::I32(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::I64(p) => {
                TypedRasterQueryProcessor::I64(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::F32(p) => {
                TypedRasterQueryProcessor::F32(KmeansPyProcessor::new(p).boxed())
            }
            TypedRasterQueryProcessor::F64(p) => {
                TypedRasterQueryProcessor::F64(KmeansPyProcessor::new(p).boxed())
            }
        })
    }
}

pub struct KmeansPyProcessor<T>
where
    T: Pixel,
{
    raster: Box<dyn RasterQueryProcessor<RasterType = T>>,
    pymod_kmeans: Py<PyModule>,
}

impl<T> KmeansPyProcessor<T>
where
    T: Pixel + numpy::Element,
    //         ^^^^^^^^^^^^^^
    // neccessary because of array transfer to python
{
    pub fn new(raster: Box<dyn RasterQueryProcessor<RasterType = T>>) -> Self {
        // temporary py stuff
        let gil = Python::acquire_gil();
        let py = gil.python();

        // saving your python script file as a struct field
        // using this, we can access python functions and objects without loss of memory state
        // on successive iterations
        let py_mdl_kmeans: Py<PyModule> =
            PyModule::from_code(py, include_str!("kmeans.py"), "py_kmeans.py", "py_kmeans")
                .unwrap()
                .into_py(py);

        Self {
            raster,
            pymod_kmeans: py_mdl_kmeans,
        }
    }

    /// Returns a new tile with data of the change map
    ///
    /// # Arguments
    ///
    /// * 'tile_pre' - Tile with older timestamp
    /// * 'tile_post' - Tile with newer timestamp
    fn kmeans(
        &self,
        tile_pre: RasterTile2D<T>,
        tile_post: RasterTile2D<T>,
    ) -> Result<RasterTile2D<T>> {
        let data_pre: Vec<T> = tile_pre.grid_array.data.clone();
        let data_post: Vec<T> = tile_post.grid_array.data.clone();

        let tile_size = tile_post.grid_array.shape.shape_array;

        // applying some steps to make data python compatible
        let arr_pre: ndarray::Array2<T> =
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_pre.to_owned())
                .unwrap()
                .to_owned();

        let arr_post: ndarray::Array2<T> =
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_post.to_owned())
                .unwrap()
                .to_owned();

        let gil = Python::acquire_gil();
        let py = gil.python();
        let pythonized_data_pre = PyArray2::from_owned_array(py, arr_pre);
        let pythonized_data_post = PyArray2::from_owned_array(py, arr_post);

        // call python algorihm and receive computation results as new tile
        let changemap_tile = self
            .pymod_kmeans
            .as_ref(py)
            .call(
                "find_PCAKmeans",
                (pythonized_data_pre, pythonized_data_post),
                None,
            )
            .unwrap()
            .downcast::<PyArray2<T>>()
            .unwrap()
            .to_vec()
            .unwrap();

        Ok(RasterTile2D::new(
            tile_pre.time,
            tile_pre.tile_position,
            tile_pre.geo_transform(),
            Grid2D::new(
                tile_pre.grid_array.shape,
                changemap_tile,
                tile_pre.grid_array.no_data_value,
            )?,
        ))
    }
}

impl<T> RasterQueryProcessor for KmeansPyProcessor<T>
where
    T: Pixel + numpy::Element,
{
    type RasterType = T;

    fn raster_query<'a>(
        &'a self,
        query: QueryRectangle,
        ctx: &'a dyn QueryContext,
    ) -> Result<BoxStream<'a, Result<RasterTile2D<Self::RasterType>>>> {
        // setting up two different points in time to compare

        let time_interval_pre = TimeInterval::new(
            NaiveDate::from_ymd(2014, 1, 1).and_hms(0, 0, 0),
            NaiveDate::from_ymd(2014, 1, 1).and_hms(0, 0, 0),
        )
        .unwrap();

        let time_interval_post = TimeInterval::new(
            NaiveDate::from_ymd(2014, 6, 1).and_hms(0, 0, 0),
            NaiveDate::from_ymd(2014, 6, 1).and_hms(0, 0, 0),
        )
        .unwrap();

        let qrect_pre = QueryRectangle {
            bbox: query.bbox,
            time_interval: time_interval_pre,
            spatial_resolution: query.spatial_resolution,
        };

        let qrect_post = QueryRectangle {
            bbox: query.bbox,
            time_interval: time_interval_post,
            spatial_resolution: query.spatial_resolution,
        };

        // generate streams for both timestamps
        let stream_pre = self.raster.query(qrect_pre, ctx)?;

        let stream_post = self.raster.query(qrect_post, ctx)?;

        // zip streams and apply python algorithm on pairwise tiles
        Ok(stream_pre
            .zip(stream_post)
            .map(move |(rt_pre, rt_post)| self.kmeans(rt_pre.unwrap(), rt_post.unwrap()))
            .boxed())
    }
}
