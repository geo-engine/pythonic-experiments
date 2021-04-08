use chrono::NaiveDate;
use futures::StreamExt;
use geoengine_datatypes::{
    dataset::{DataSetId, InternalDataSetId},
    primitives::{SpatialResolution, TimeInterval},
};
use geoengine_datatypes::{operations::image::ToPng, raster::Blit};
use geoengine_datatypes::{
    operations::image::{Colorizer, RgbaColor},
    spatial_reference::SpatialReference,
};
use geoengine_datatypes::{
    primitives::{BoundingBox2D, Measurement, TimeGranularity, TimeStep},
    raster::Pixel,
};
use geoengine_datatypes::{
    raster::{GeoTransform, Grid2D, RasterDataType, RasterTile2D},
    util::Identifier,
};
use geoengine_operators::engine::{
    MockExecutionContext, MockQueryContext, QueryContext, QueryRectangle, RasterOperator,
    RasterQueryProcessor, RasterResultDescriptor,
};
use geoengine_operators::source::{
    FileNotFoundHandling, GdalDataSetParameters, GdalMetaDataRegular, GdalSource,
    GdalSourceParameters,
};
use geoengine_services::error::Result;
use pythonic_experiments::example_pyop::{PyOperator, PyOperatorParams};
use std::{convert::TryInto, fs::File, io::Write};

#[tokio::main]
async fn main() {
    // 1. register source dataset

    let mut execution_context = MockExecutionContext::default();
    let dataset_id = DataSetId::Internal(InternalDataSetId::new());
    execution_context.add_meta_data(dataset_id.clone(), Box::new(create_ndvi_meta_data()));

    // 2. define your workflow

    let operator = PyOperator {
        params: PyOperatorParams { n_comp: 5. },
        raster_sources: vec![GdalSource {
            params: GdalSourceParameters {
                data_set: dataset_id,
            },
        }
        .boxed()],
        vector_sources: vec![],
    }
    .boxed();

    // 3. initialize operator and create query processor of correct result type, e.g., U8 raster

    let initialized_operator = operator.initialize(&execution_context).unwrap();

    let query_processor = initialized_operator
        .query_processor()
        .unwrap()
        .get_u8()
        .unwrap();

    let query_ctx = MockQueryContext::default();

    // 4. define your output image size (in px)

    let request = Request::new(1024, 512);
    // let request = Request::new(767, 510);

    // 5. define your query

    // …spatial
    let bbox = BoundingBox2D::new((-180., -90.).into(), (180., 90.).into()).unwrap();
    // let bbox = BoundingBox2D::new(
    //     (-802607.85, 3577980.78).into(),
    //     (1498701.38, 5108186.39).into(),
    // )
    // .unwrap();

    // …temporal
    let time_interval = TimeInterval::new(
        NaiveDate::from_ymd(2014, 6, 1).and_hms(0, 0, 0),
        NaiveDate::from_ymd(2014, 6, 1).and_hms(0, 0, 0),
    )
    .unwrap();

    // …and with a resolution (must fit to requested image size)
    let spatial_resolution = SpatialResolution::new(
        bbox.size_x() / f64::from(request.width),
        bbox.size_y() / f64::from(request.height),
    )
    .unwrap();

    // put this together into one struct
    let query_rect = QueryRectangle {
        bbox,
        time_interval,
        spatial_resolution,
    };

    // 6. define how to colorize the output image

    let colorizer = Colorizer::linear_gradient(
        vec![
            (0., RgbaColor::white()).try_into().unwrap(),
            (255., RgbaColor::black()).try_into().unwrap(),
        ],
        RgbaColor::transparent(),
        RgbaColor::transparent(),
    )
    .unwrap();

    // 7. collect the whole stream of raster tiles into one PNG

    let png =
        raster_stream_to_png_bytes(query_processor, query_rect, query_ctx, request, colorizer)
            .await
            .unwrap();

    // 8. store png

    let mut file = File::create("output.png").unwrap();
    file.write_all(&png).unwrap();
}

fn create_ndvi_meta_data() -> GdalMetaDataRegular {
    GdalMetaDataRegular {
        start: 1_388_534_400_000.into(),
        step: TimeStep {
            granularity: TimeGranularity::Months,
            step: 1,
        },
        placeholder: "%%%_START_TIME_%%%".to_string(),
        time_format: "%Y-%m-%d".to_string(),
        params: GdalDataSetParameters {
            file_path: "data/modis_ndvi/MOD13A2_M_NDVI_%%%_START_TIME_%%%.TIFF".into(),
            // file_path: "data/nas/mas_europe_2011_1200/20110601_1200.TIFF".into(),
            rasterband_channel: 1,
            geo_transform: GeoTransform {
                origin_coordinate: (-180., 90.).into(),
                // origin_coordinate: (-802607.84, 5108186.38).into(),
                x_pixel_size: 0.1,
                // x_pixel_size: 3000.4,
                y_pixel_size: -0.1,
                // y_pixel_size: -3000.4,
            },
            bbox: BoundingBox2D::new((-180., -90.).into(), (180., 90.).into()).unwrap(),
            // bbox: BoundingBox2D::new(
            //     (-802607.85, 3577980.78).into(),
            //     (1498701.38, 5108186.39).into(),
            // )
            // .unwrap(),
            file_not_found_handling: FileNotFoundHandling::NoData,
            no_data_value: Some(0.),
        },
        result_descriptor: RasterResultDescriptor {
            data_type: RasterDataType::U8,
            spatial_reference: SpatialReference::epsg_4326().into(),
            measurement: Measurement::Unitless,
        },
    }
}

struct Request {
    pub width: u32,
    pub height: u32,
    pub time: Option<TimeInterval>,
}

impl Request {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            time: None,
        }
    }
}

async fn raster_stream_to_png_bytes<T, C: QueryContext>(
    processor: Box<dyn RasterQueryProcessor<RasterType = T>>,
    query_rect: QueryRectangle,
    query_ctx: C,
    request: Request,
    colorizer: Colorizer,
) -> Result<Vec<u8>>
where
    T: Pixel,
{
    let tile_stream = processor.raster_query(query_rect, &query_ctx)?;

    let x_query_resolution = query_rect.bbox.size_x() / f64::from(request.width);
    let y_query_resolution = query_rect.bbox.size_y() / f64::from(request.height);

    // build png
    let dim = [request.height as usize, request.width as usize];
    let query_geo_transform = GeoTransform::new(
        query_rect.bbox.upper_left(),
        x_query_resolution,
        -y_query_resolution,
    );

    let output_raster = Grid2D::new_filled(dim.into(), T::zero(), None);
    let output_tile = Ok(RasterTile2D::new_without_offset(
        request.time.unwrap_or_default(),
        query_geo_transform,
        output_raster,
    ));

    let output_tile = tile_stream
        .fold(output_tile, |raster2d, tile| {
            let result: Result<RasterTile2D<T>> = match (raster2d, tile) {
                (Ok(mut raster2d), Ok(tile)) => match raster2d.blit(tile) {
                    // ! was ist blit?
                    Ok(_) => Ok(raster2d),
                    Err(error) => Err(error.into()),
                },
                (Err(error), _) => Err(error),
                (_, Err(error)) => Err(error.into()),
            };

            match result {
                Ok(updated_raster2d) => futures::future::ok(updated_raster2d),
                Err(error) => futures::future::err(error),
            }
        })
        .await?;

    Ok(output_tile.to_png(request.width, request.height, &colorizer)?)
}
