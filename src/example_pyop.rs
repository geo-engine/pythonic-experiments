use std::convert::TryInto;

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

use ndarray::{s, stack, Array, Array1, Array2, Axis, Dim, OwnedArcRepr};
use numpy::{IntoPyArray, PyArray, PyArray2, ToPyArray};
use pyo3::types::{PyAny, PyModule};

/// An example operator that adds `x` to its input raster stream
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PyOperator {
    pub params: PyOperatorParams,
    pub raster_sources: Vec<Box<dyn RasterOperator>>,
    pub vector_sources: Vec<Box<dyn VectorOperator>>,
}

/// The parameter spec for `PyOperator`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PyOperatorParams {
    /// Number of components for PCA
    pub n_comp: f64,
}

#[typetag::serde]
impl RasterOperator for PyOperator {
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
            params: self.params,
            raster_sources: vec![initialized_raster],
            vector_sources: vec![],
            result_descriptor,
            state: (),
        };

        Ok(initialized_operator.boxed())
    }
}

pub struct InitializedPyOperator {
    pub params: PyOperatorParams,
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
        let add_value = self.params.n_comp;

        Ok(match typed_raster_processor {
            TypedRasterQueryProcessor::U8(p) => {
                TypedRasterQueryProcessor::U8(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::U16(p) => {
                TypedRasterQueryProcessor::U16(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::U32(p) => {
                TypedRasterQueryProcessor::U32(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::U64(p) => {
                TypedRasterQueryProcessor::U64(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::I8(p) => {
                TypedRasterQueryProcessor::I8(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::I16(p) => {
                TypedRasterQueryProcessor::I16(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::I32(p) => {
                TypedRasterQueryProcessor::I32(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::I64(p) => {
                TypedRasterQueryProcessor::I64(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::F32(p) => {
                TypedRasterQueryProcessor::F32(PyProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::F64(p) => {
                TypedRasterQueryProcessor::F64(PyProcessor::new(p, add_value).boxed())
            }
        })
    }
}

pub struct PyProcessor<T>
where
    T: Pixel,
{
    raster: Box<dyn RasterQueryProcessor<RasterType = T>>,
    add_value: T,
}

impl<T> PyProcessor<T>
where
    T: Pixel + numpy::Element,
{
    pub fn new(raster: Box<dyn RasterQueryProcessor<RasterType = T>>, add_value: f64) -> Self {
        Self {
            raster,
            add_value: T::from_(add_value),
        }
    }

    fn compute_py(&self, tile: RasterTile2D<T>) -> Result<RasterTile2D<T>> {
        //register GIL
        let gil = pyo3::Python::acquire_gil();

        //create a Pyhton object, which represents an instance of a Python interpreter
        let py = gil.python();

        let py_ml: &PyModule = PyModule::from_code(
            py,
            include_str!("simple_pca.py"),
            "filename.py",
            "modulename",
        )
        .unwrap();

        // source tile data
        let data = &tile.grid_array.data;
        let subarr: Array2<T> = Array2::from_shape_vec((4, 4), data.to_owned())
            .unwrap()
            .to_owned();
        let pythonized_data = PyArray2::from_owned_array(py, subarr);

        let pca_res = py_ml.call("run_pca", (2, pythonized_data), None).unwrap();

        let new_data: Vec<T> = pca_res
            // .get_item(0)
            // .unwrap()
            .downcast::<PyArray2<T>>()
            .unwrap()
            .to_vec()
            .unwrap();
        // .to_owned_array();

        // manipulate data
        //                     (vvv--- hier wird nur nach vollständigen daten unterschieden---)
        // let new_data: Vec<T> = data.iter().map(|&v| v + self.add_value).collect();

        // return raster tile with manipulated data
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

    fn compute(&self, tile: RasterTile2D<T>) -> Result<RasterTile2D<T>> {
        // source tile data
        let data: &[T] = &tile.grid_array.data;

        // manipulate data
        //                     (vvv--- hier wird nur nach vollständigen daten unterschieden---)
        let new_data: Vec<T> = if let Some(no_data_value) = tile.grid_array.no_data_value {
            data.iter()
                .map(|&v| {
                    // TODO: what about v + x = no data?
                    if v == no_data_value {
                        no_data_value
                    } else {
                        v + self.add_value
                    }
                })
                .collect()
        } else {
            // ! welchen zweck erfüllt dieser block?
            data.iter().map(|&v| v + self.add_value).collect()
        };

        // return raster tile with manipulated data
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

impl<T> RasterQueryProcessor for PyProcessor<T>
where
    T: Pixel + numpy::Element,
{
    type RasterType = T;

    // ! wie wird diese funktion getriggert?
    fn raster_query<'a>(
        &'a self,
        query: QueryRectangle,
        ctx: &'a dyn QueryContext,
    ) -> Result<BoxStream<'a, Result<RasterTile2D<Self::RasterType>>>> {
        Ok(self
            .raster
            .query(query, ctx)? // ? hier werden einzelne kacheln aus dem ursprungs raster als neue subraster prozessiert
            .map(move |raster_tile| {
                let raster_tile = raster_tile?;
                // self.compute(raster_tile) // * hier wird der operator angewendet
                self.compute_py(raster_tile)
            })
            .boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geoengine_datatypes::primitives::{
        BoundingBox2D, Measurement, SpatialResolution, TimeInterval,
    };
    use geoengine_datatypes::raster::{RasterDataType, TileInformation};
    use geoengine_datatypes::spatial_reference::SpatialReference;
    use geoengine_operators::engine::{MockExecutionContext, MockQueryContext};
    use geoengine_operators::mock::{MockRasterSource, MockRasterSourceParams};

    #[tokio::test]
    async fn simple_raster() {
        // ! Frage: kompilierzeit verkürzen?
        // ! Frage: wie sind die tiles orientiert, bzw. wie werden sie prozessiert? -> zeilenweise links nach rechts, unten nach oben?
        // ausgangs raster tile (pre-state)
        let raster_tile = RasterTile2D::new_with_tile_info(
            TimeInterval::default(),
            TileInformation {
                global_geo_transform: Default::default(),
                global_tile_position: [0, 0].into(),
                tile_size_in_pixels: [4, 4].into(), // ? y = -1?
            },
            Grid2D::new(
                [4, 4].into(),
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                None,
            )
            .unwrap(),
        );

        // kontext faktoren. nötig, aber hier eigentlich nicht weiter relevant?
        // -> liefert aber auch die eigentlichen daten, die bearbeitet werden sollen.
        let raster_source = MockRasterSource {
            params: MockRasterSourceParams {
                data: vec![raster_tile], // ! hier liegen die daten
                result_descriptor: RasterResultDescriptor {
                    data_type: RasterDataType::U8,
                    spatial_reference: SpatialReference::epsg_4326().into(),
                    measurement: Measurement::Unitless,
                },
            },
        }
        .boxed();

        let operator = PyOperator {
            params: PyOperatorParams { n_comp: 1. },
            raster_sources: vec![raster_source],
            vector_sources: vec![],
        };

        let execution_context = MockExecutionContext::default();

        // * hier kommt ein InitializedADDXOperator raus
        let operator = operator.boxed().initialize(&execution_context).unwrap();
        let query_processor = operator.query_processor().unwrap().get_u8().unwrap();

        let result = query_processor
            .query(
                QueryRectangle {
                    bbox: BoundingBox2D::new((0.0, 0.0).into(), (2.0, 2.0).into()).unwrap(),
                    time_interval: Default::default(),
                    spatial_resolution: SpatialResolution::new(1., 1.).unwrap(),
                },
                &MockQueryContext::new(0),
            )
            .unwrap()
            .map(|tile| tile.unwrap())
            .collect::<Vec<_>>()
            .await;

        let result_tile = RasterTile2D::new_with_tile_info(
            TimeInterval::default(),
            TileInformation {
                global_geo_transform: Default::default(),
                global_tile_position: [0, 0].into(),
                tile_size_in_pixels: [4, 4].into(),
            },
            Grid2D::new(
                [4, 4].into(),
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                None,
            )
            .unwrap(),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], result_tile);
    }
}
