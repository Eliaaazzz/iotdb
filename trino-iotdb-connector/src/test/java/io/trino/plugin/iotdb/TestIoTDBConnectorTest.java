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
import io.trino.plugin.jdbc.BaseJdbcConnectorTest;
import io.trino.testing.QueryRunner;
import io.trino.testing.TestingConnectorBehavior;
import io.trino.testing.sql.SqlExecutor;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.junit.jupiter.api.TestInstance.Lifecycle.PER_CLASS;

/**
 * Integration test extending BaseJdbcConnectorTest.
 * Inherits 400+ standard Trino JDBC connector tests.
 * <p>
 * Requires Docker to run (uses Testcontainers with IoTDB image).
 * Tests that IoTDB does not support (writes, DDL, transactions) are
 * overridden to skip or expect NOT_SUPPORTED.
 */
@TestInstance(PER_CLASS)
class TestIoTDBConnectorTest
        extends BaseJdbcConnectorTest
{
    private TestingIoTDBServer server;

    @Override
    protected QueryRunner createQueryRunner()
            throws Exception
    {
        server = new TestingIoTDBServer();
        return IoTDBQueryRunner.createIoTDBQueryRunner(server, ImmutableMap.of());
    }

    @AfterAll
    public void tearDown()
    {
        if (server != null) {
            server.close();
            server = null;
        }
    }

    @Override
    protected SqlExecutor onRemoteDatabase()
    {
        return sql -> {
            try {
                server.execute(sql);
            }
            catch (Exception e) {
                throw new RuntimeException("Failed to execute on IoTDB: " + sql, e);
            }
        };
    }

    // IoTDB does not support write operations via this connector
    @Override
    protected boolean hasBehavior(TestingConnectorBehavior connectorBehavior)
    {
        switch (connectorBehavior) {
            case SUPPORTS_CREATE_TABLE:
            case SUPPORTS_CREATE_TABLE_WITH_DATA:
            case SUPPORTS_INSERT:
            case SUPPORTS_DELETE:
            case SUPPORTS_UPDATE:
            case SUPPORTS_MERGE:
            case SUPPORTS_RENAME_TABLE:
            case SUPPORTS_RENAME_COLUMN:
            case SUPPORTS_ADD_COLUMN:
            case SUPPORTS_DROP_COLUMN:
            case SUPPORTS_SET_COLUMN_TYPE:
            case SUPPORTS_RENAME_TABLE_ACROSS_SCHEMAS:
            case SUPPORTS_CREATE_SCHEMA:
            case SUPPORTS_DROP_SCHEMA_CASCADE:
            case SUPPORTS_RENAME_SCHEMA:
            case SUPPORTS_CREATE_VIEW:
            case SUPPORTS_CREATE_MATERIALIZED_VIEW:
            case SUPPORTS_COMMENT_ON_TABLE:
            case SUPPORTS_COMMENT_ON_COLUMN:
                return false;
            default:
                return super.hasBehavior(connectorBehavior);
        }
    }

    @Test
    public void testShowSchemas()
    {
        assertQuery("SHOW SCHEMAS FROM iotdb", "VALUES 'test_db'");
    }
}
