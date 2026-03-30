/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.trino.plugin.iotdb;

import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.utility.DockerImageName;

import java.io.Closeable;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.Duration;
import java.util.Properties;

/**
 * Testcontainers wrapper for a standalone Apache IoTDB instance.
 * Uses the official IoTDB Docker image with Table Model (sql_dialect=table).
 */
public class TestingIoTDBServer
        implements Closeable
{
    private static final String IOTDB_IMAGE = "apache/iotdb:2.0.6-standalone";
    private static final int IOTDB_RPC_PORT = 6667;

    private final GenericContainer<?> container;

    public TestingIoTDBServer()
    {
        container = new GenericContainer<>(DockerImageName.parse(IOTDB_IMAGE))
                .withExposedPorts(IOTDB_RPC_PORT)
                .waitingFor(Wait.forListeningPort())
                .withStartupTimeout(Duration.ofMinutes(3));
        container.start();
    }

    /**
     * Get JDBC URL for the IoTDB container with Table Model dialect.
     */
    public String getJdbcUrl()
    {
        return String.format("jdbc:iotdb://%s:%d?sql_dialect=table",
                container.getHost(),
                container.getMappedPort(IOTDB_RPC_PORT));
    }

    /**
     * Get the mapped RPC port.
     */
    public int getPort()
    {
        return container.getMappedPort(IOTDB_RPC_PORT);
    }

    /**
     * Get the container host.
     */
    public String getHost()
    {
        return container.getHost();
    }

    /**
     * Execute SQL on the IoTDB container directly (for test data setup).
     */
    public void execute(String sql)
            throws SQLException
    {
        Properties properties = new Properties();
        properties.setProperty("user", "root");
        properties.setProperty("password", "root");
        properties.setProperty("sql_dialect", "table");

        try (Connection connection = DriverManager.getConnection(
                String.format("jdbc:iotdb://%s:%d",
                        container.getHost(),
                        container.getMappedPort(IOTDB_RPC_PORT)),
                properties);
                Statement statement = connection.createStatement()) {
            statement.execute(sql);
        }
    }

    @Override
    public void close()
    {
        container.stop();
    }
}
