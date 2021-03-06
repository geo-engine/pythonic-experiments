pub mod error;
// pub mod example_operator;
pub mod example_pyop;

#[cfg(test)]
mod tests {
    use geoengine_services::error::Error;
    use geoengine_services::error::Result;
    use geoengine_services::server::start_server;
    use tokio::sync::oneshot;

    /// Start the Geo Engine
    #[tokio::test]
    async fn start_geo_engine() -> Result<(), Error> {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let server = start_server(Some(shutdown_rx), None);

        shutdown_tx.send(()).unwrap();

        server.await
    }
}
