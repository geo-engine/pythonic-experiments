pub mod example_operator;
pub mod error;

#[cfg(test)]
mod tests {
    use tokio::sync::oneshot;
    use geoengine_services::server::start_server;
    use geoengine_services::error::Error;

    /// Start the Geo Engine
    #[tokio::test]
    async fn start_geo_engine() -> Result<(), Error> {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let server = start_server(Some(shutdown_rx), None);

        shutdown_tx.send(()).unwrap();

        server.await
    }
}
