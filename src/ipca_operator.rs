use futures::stream::BoxStream;
use futures::StreamExt;
use geoengine_datatypes::raster::{Grid2D, Pixel, Raster, RasterTile2D};
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

/// An example operator that runs a compression using ipca
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IpcaOperator {
    pub params: IpcaOperatorParams,
    pub raster_sources: Vec<Box<dyn RasterOperator>>,
    pub vector_sources: Vec<Box<dyn VectorOperator>>,
}

/// The parameter spec for `PyOperator`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IpcaOperatorParams {
    /// Number of components for PCA
    pub n_comp: usize,
}

#[typetag::serde]
impl RasterOperator for IpcaOperator {
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

        let initialized_operator = InitializedIpcaOperator {
            params: self.params,
            raster_sources: vec![initialized_raster],
            vector_sources: vec![],
            result_descriptor,
            state: (),
        };

        Ok(initialized_operator.boxed())
    }
}

pub struct InitializedIpcaOperator {
    pub params: IpcaOperatorParams,
    pub raster_sources: Vec<Box<InitializedRasterOperator>>,
    pub vector_sources: Vec<Box<InitializedVectorOperator>>,
    pub result_descriptor: RasterResultDescriptor,
    pub state: (),
}

impl InitializedOperatorBase for InitializedIpcaOperator {
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
    for InitializedIpcaOperator
{
    fn query_processor(&self) -> Result<TypedRasterQueryProcessor> {
        let typed_raster_processor = self.raster_sources[0].query_processor()?;
        let n_comp = self.params.n_comp;

        Ok(match typed_raster_processor {
            TypedRasterQueryProcessor::U8(p) => {
                TypedRasterQueryProcessor::U8(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::U16(p) => {
                TypedRasterQueryProcessor::U16(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::U32(p) => {
                TypedRasterQueryProcessor::U32(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::U64(p) => {
                TypedRasterQueryProcessor::U64(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::I8(p) => {
                TypedRasterQueryProcessor::I8(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::I16(p) => {
                TypedRasterQueryProcessor::I16(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::I32(p) => {
                TypedRasterQueryProcessor::I32(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::I64(p) => {
                TypedRasterQueryProcessor::I64(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::F32(p) => {
                TypedRasterQueryProcessor::F32(IpcaPyProcessor::new(p, n_comp).boxed())
            }
            TypedRasterQueryProcessor::F64(p) => {
                TypedRasterQueryProcessor::F64(IpcaPyProcessor::new(p, n_comp).boxed())
            }
        })
    }
}

pub struct IpcaPyProcessor<T>
where
    T: Pixel,
{
    raster: Box<dyn RasterQueryProcessor<RasterType = T>>,
    pymod_ipca: Py<PyModule>,
    n_components: usize,
}

impl<T> IpcaPyProcessor<T>
where
    T: Pixel + numpy::Element,
    //         ^^^^^^^^^^^^^^
    // neccessary because of array transfer to python
{
    pub fn new(raster: Box<dyn RasterQueryProcessor<RasterType = T>>, n_comp: usize) -> Self {
        // temporary py stuff
        let gil = Python::acquire_gil();
        let py = gil.python();

        // saving your python script file as a struct field
        // using this, we can access python functions and objects without loss of memory state
        // on successive iterations
        let py_mdl_ipca: Py<PyModule> =
            PyModule::from_code(py, include_str!("ipca.py"), "py_ipca.py", "py_ipca")
                .unwrap()
                .into_py(py);

        Self {
            raster,
            pymod_ipca: py_mdl_ipca,
            n_components: n_comp,
        }
    }

    /// Initialize a new IPCA instance in python
    fn initialize_ipca(&self) {
        let gil = Python::acquire_gil();
        let py = gil.python();

        self.pymod_ipca
            .as_ref(py)
            .call("init", (self.n_components,), None)
            .expect("something went wrong with initializing ipca object");
    }

    /// Sends tile to IPCA instance in python. Data will be fitted.
    ///
    /// # Arguments
    ///
    /// * 'tile' - Tile to be fitted

    fn fit_tiles(&self, tile: RasterTile2D<T>) -> Result<RasterTile2D<T>> {
        let tile_size = tile.grid_array.shape.shape_array;

        let data: Vec<T> = tile.grid_array.data.clone();

        // preparing data for python
        let arr: ndarray::Array2<T> =
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data.to_owned())
                .unwrap()
                .to_owned();

        let gil = Python::acquire_gil();
        let py = gil.python();
        let pythonized_data = PyArray2::from_owned_array(py, arr);

        // calling python
        self.pymod_ipca
            .as_ref(py)
            .call("partial_fit_ipca", (pythonized_data,), None)
            .expect("something went wrong with fitting the tile");

        // todo: diese rückgabe ist eigentlich unnötig
        Ok(RasterTile2D::new(
            tile.time,
            tile.tile_position,
            tile.geo_transform(),
            Grid2D::new(tile.grid_array.shape, data, tile.grid_array.no_data_value)?,
        ))
    }

    /// Returns a new tile with transformed data from python
    ///
    /// # Arguments
    ///
    /// * 'tile' - Tile to be transformed
    fn transform_tiles(&self, tile: RasterTile2D<T>) -> Result<RasterTile2D<T>> {
        let tile_size = tile.grid_array.shape.shape_array;

        let data: Vec<T> = tile.grid_array.data.clone();

        // preparing data for python
        let arr: ndarray::Array2<T> =
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data.to_owned())
                .unwrap()
                .to_owned();

        let gil = Python::acquire_gil();
        let py = gil.python();
        let pythonized_data = PyArray2::from_owned_array(py, arr);

        // calling python
        let new_data = self
            .pymod_ipca
            .as_ref(py)
            .call("apply_ipca", (pythonized_data,), None)
            .unwrap()
            .downcast::<PyArray2<T>>()
            .unwrap()
            .to_vec()
            .unwrap();

        Ok(RasterTile2D::new(
            tile.time,
            tile.tile_position,
            tile.geo_transform(),
            Grid2D::new(
                tile.grid_array.shape,
                new_data,
                tile.grid_array.no_data_value,
            )?,
        ))
    }
}

impl<T> RasterQueryProcessor for IpcaPyProcessor<T>
where
    T: Pixel + numpy::Element,
{
    type RasterType = T;

    fn raster_query<'a>(
        &'a self,
        query: QueryRectangle,
        ctx: &'a dyn QueryContext,
    ) -> Result<BoxStream<'a, Result<RasterTile2D<Self::RasterType>>>> {
        self.initialize_ipca();

        // first stream is only used to fit tiles
        let s1 = self.raster.query(query, ctx)?.map(move |raster_tile| {
            let raster_tile = raster_tile.unwrap();

            self.fit_tiles(raster_tile)
        });

        // second stream is used to get transformed data
        let s2 = self.raster.query(query, ctx)?.map(move |raster_tile| {
            let raster_tile = raster_tile.unwrap();

            self.transform_tiles(raster_tile)
        });

        // sequentially execute streams
        let res = s1.chain(s2).boxed();
        Ok(res)
    }
}
