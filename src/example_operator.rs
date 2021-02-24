use crate::error::Result;
use futures::stream::BoxStream;
use futures::StreamExt;
use geoengine_datatypes::raster::{Grid2D, Pixel, Raster, RasterTile2D};
use geoengine_operators::engine::{
    ExecutionContext, InitializedOperator, InitializedOperatorBase, InitializedRasterOperator,
    InitializedVectorOperator, QueryContext, QueryProcessor, QueryRectangle, RasterOperator,
    RasterQueryProcessor, RasterResultDescriptor, TypedRasterQueryProcessor, VectorOperator,
};
use geoengine_operators::error::Error as GeoengineOperatorsError;
use serde::{Deserialize, Serialize};

/// An example operator that adds `x` to its input raster stream
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AddXOperator {
    pub params: AddXOperatorParams,
    pub raster_sources: Vec<Box<dyn RasterOperator>>,
    pub vector_sources: Vec<Box<dyn VectorOperator>>,
}

/// The parameter spec for `AddXOperator`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AddXOperatorParams {
    /// A value to add to its input raster stream
    pub x: f64,
}

#[typetag::serde]
impl RasterOperator for AddXOperator {
    fn initialize(
        mut self: Box<Self>,
        context: &dyn ExecutionContext,
    ) -> Result<Box<InitializedRasterOperator>, GeoengineOperatorsError> {
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

        let initialized_operator = InitializedAddXOperator {
            params: self.params,
            raster_sources: vec![initialized_raster],
            vector_sources: vec![],
            result_descriptor,
            state: (),
        };

        Ok(initialized_operator.boxed())
    }
}

pub struct InitializedAddXOperator {
    pub params: AddXOperatorParams,
    pub raster_sources: Vec<Box<InitializedRasterOperator>>,
    pub vector_sources: Vec<Box<InitializedVectorOperator>>,
    pub result_descriptor: RasterResultDescriptor,
    pub state: (),
}

impl InitializedOperatorBase for InitializedAddXOperator {
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
    for InitializedAddXOperator
{
    fn query_processor(&self) -> Result<TypedRasterQueryProcessor, GeoengineOperatorsError> {
        let typed_raster_processor = self.raster_sources[0].query_processor()?;
        let add_value = self.params.x;

        Ok(match typed_raster_processor {
            TypedRasterQueryProcessor::U8(p) => {
                TypedRasterQueryProcessor::U8(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::U16(p) => {
                TypedRasterQueryProcessor::U16(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::U32(p) => {
                TypedRasterQueryProcessor::U32(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::U64(p) => {
                TypedRasterQueryProcessor::U64(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::I8(p) => {
                TypedRasterQueryProcessor::I8(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::I16(p) => {
                TypedRasterQueryProcessor::I16(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::I32(p) => {
                TypedRasterQueryProcessor::I32(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::I64(p) => {
                TypedRasterQueryProcessor::I64(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::F32(p) => {
                TypedRasterQueryProcessor::F32(AddXProcessor::new(p, add_value).boxed())
            }
            TypedRasterQueryProcessor::F64(p) => {
                TypedRasterQueryProcessor::F64(AddXProcessor::new(p, add_value).boxed())
            }
        })
    }
}

pub struct AddXProcessor<T: Pixel> {
    raster: Box<dyn RasterQueryProcessor<RasterType = T>>,
    add_value: T,
}

impl<T: Pixel> AddXProcessor<T> {
    pub fn new(raster: Box<dyn RasterQueryProcessor<RasterType = T>>, add_value: f64) -> Self {
        Self {
            raster,
            add_value: T::from_(add_value),
        }
    }

    fn compute(&self, tile: RasterTile2D<T>) -> Result<RasterTile2D<T>, GeoengineOperatorsError> {
        let data: &[T] = &tile.grid_array.data;

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
            data.iter().map(|&v| v + self.add_value).collect()
        };

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

impl<T: Pixel> RasterQueryProcessor for AddXProcessor<T> {
    type RasterType = T;

    fn raster_query<'a>(
        &'a self,
        query: QueryRectangle,
        ctx: &'a dyn QueryContext,
    ) -> BoxStream<'a, Result<RasterTile2D<Self::RasterType>, GeoengineOperatorsError>> {
        self.raster
            .query(query, ctx)
            .map(move |raster_tile| {
                let raster_tile = raster_tile?;
                self.compute(raster_tile)
            })
            .boxed()
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
        let raster_tile = RasterTile2D::new_with_tile_info(
            TimeInterval::default(),
            TileInformation {
                global_geo_transform: Default::default(),
                global_tile_position: [0, 0].into(),
                tile_size_in_pixels: [3, 2].into(),
            },
            Grid2D::new([3, 2].into(), vec![1, 2, 3, 4, 5, 6], None).unwrap(),
        );

        let raster_source = MockRasterSource {
            params: MockRasterSourceParams {
                data: vec![raster_tile],
                result_descriptor: RasterResultDescriptor {
                    data_type: RasterDataType::U8,
                    spatial_reference: SpatialReference::epsg_4326().into(),
                    measurement: Measurement::Unitless,
                },
            },
        }
        .boxed();

        let operator = AddXOperator {
            params: AddXOperatorParams { x: 1. },
            raster_sources: vec![raster_source],
            vector_sources: vec![],
        };

        let execution_context = MockExecutionContext::default();

        let operator = operator.boxed().initialize(&execution_context).unwrap();
        let query_processor = operator.query_processor().unwrap().get_u8().unwrap();

        let result = query_processor
            .query(
                QueryRectangle {
                    bbox: BoundingBox2D::new((0.0, 0.0).into(), (3.0, 2.0).into()).unwrap(),
                    time_interval: Default::default(),
                    spatial_resolution: SpatialResolution::new(1., 1.).unwrap(),
                },
                &MockQueryContext::new(0),
            )
            .map(|tile| tile.unwrap())
            .collect::<Vec<_>>()
            .await;

        let result_tile = RasterTile2D::new_with_tile_info(
            TimeInterval::default(),
            TileInformation {
                global_geo_transform: Default::default(),
                global_tile_position: [0, 0].into(),
                tile_size_in_pixels: [3, 2].into(),
            },
            Grid2D::new([3, 2].into(), vec![2, 3, 4, 5, 6, 7], None).unwrap(),
        );

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], result_tile);
    }
}
