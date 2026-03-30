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

import com.google.common.collect.ImmutableMap;
import io.trino.testing.DistributedQueryRunner;
import io.trino.testing.QueryRunner;

import java.util.Map;

import static io.trino.testing.TestingSession.testSessionBuilder;

/**
 * Factory for creating a DistributedQueryRunner with the IoTDB connector.
 * Used by integration tests to set up a Trino environment connected to an IoTDB container.
 */
public final class IoTDBQueryRunner
{
    private IoTDBQueryRunner() {}

    /**
     * Create a QueryRunner with the IoTDB plugin installed and a catalog configured
     * pointing to the given TestingIoTDBServer.
     */
    public static QueryRunner createIoTDBQueryRunner(
            TestingIoTDBServer server,
            Map<String, String> extraProperties)
            throws Exception
    {
        DistributedQueryRunner queryRunner = DistributedQueryRunner.builder(
                        testSessionBuilder()
                                .setCatalog("iotdb")
                                .setSchema("test_db")
                                .build())
                .setExtraProperties(extraProperties)
                .build();

        try {
            queryRunner.installPlugin(new IoTDBPlugin());
            queryRunner.createCatalog("iotdb", "iotdb", ImmutableMap.of(
                    "connection-url", server.getJdbcUrl(),
                    "connection-user", "root",
                    "connection-password", "root"));

            // Create test database and sample table
            setupTestData(server);

            return queryRunner;
        }
        catch (Exception e) {
            queryRunner.close();
            throw e;
        }
    }

    /**
     * Create a QueryRunner with default properties.
     */
    public static QueryRunner createIoTDBQueryRunner(TestingIoTDBServer server)
            throws Exception
    {
        return createIoTDBQueryRunner(server, ImmutableMap.of());
    }

    /**
     * Set up test database and sample table with sensor data.
     * Uses IoTDB's Table Model schema:
     * - TIME: implicit timestamp column
     * - TAG columns: region_id, device_id (indexed dimensions)
     * - ATTRIBUTE columns: model (static metadata)
     * - FIELD columns: temperature, humidity (measured values)
     */
    private static void setupTestData(TestingIoTDBServer server)
            throws Exception
    {
        server.execute("CREATE DATABASE IF NOT EXISTS test_db");
        server.execute("USE test_db");
        server.execute(
                "CREATE TABLE IF NOT EXISTS sensors (" +
                        "region_id STRING TAG, " +
                        "device_id STRING TAG, " +
                        "model STRING ATTRIBUTE, " +
                        "temperature FLOAT FIELD, " +
                        "humidity FLOAT FIELD)");

        // Insert sample data
        server.execute(
                "INSERT INTO sensors(time, region_id, device_id, model, temperature, humidity) " +
                        "VALUES (1704067200000, 'us-west', 'dev-001', 'ModelA', 22.5, 45.0)");
        server.execute(
                "INSERT INTO sensors(time, region_id, device_id, model, temperature, humidity) " +
                        "VALUES (1704067260000, 'us-west', 'dev-001', 'ModelA', 22.8, 44.5)");
        server.execute(
                "INSERT INTO sensors(time, region_id, device_id, model, temperature, humidity) " +
                        "VALUES (1704067200000, 'us-east', 'dev-002', 'ModelB', 18.3, 62.1)");
        server.execute(
                "INSERT INTO sensors(time, region_id, device_id, model, temperature, humidity) " +
                        "VALUES (1704067260000, 'us-east', 'dev-002', 'ModelB', 18.5, 61.8)");
        server.execute(
                "INSERT INTO sensors(time, region_id, device_id, model, temperature, humidity) " +
                        "VALUES (1704067320000, 'us-west', 'dev-003', 'ModelA', 25.1, 38.2)");
    }
}
