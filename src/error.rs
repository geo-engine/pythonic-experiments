use snafu::Snafu;
use std::ops::Range;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum Error {

    #[snafu(display("DatatypeError: {}", source))]
    Datatype {
        source: geoengine_datatypes::error::Error,
    },

    #[snafu(display("OperatorError: {}", source))]
    Operator {
        source: geoengine_operators::error::Error,
    },

    #[snafu(display("InvalidNumberOfRasterInputsError: expected \"[{} .. {}]\" found \"{}\"", expected.start, expected.end, found))]
    InvalidNumberOfRasterInputs {
        expected: Range<usize>,
        found: usize,
    },

    #[snafu(display("InvalidNumberOfVectorInputsError: expected \"[{} .. {}]\" found \"{}\"", expected.start, expected.end, found))]
    InvalidNumberOfVectorInputs {
        expected: Range<usize>,
        found: usize,
    },

}

impl From<geoengine_datatypes::error::Error> for Error {
    fn from(source: geoengine_datatypes::error::Error) -> Self {
        Self::Datatype { source }
    }
}

impl From<geoengine_operators::error::Error> for Error {
    fn from(source: geoengine_operators::error::Error) -> Self {
        Self::Operator { source }
    }
}
