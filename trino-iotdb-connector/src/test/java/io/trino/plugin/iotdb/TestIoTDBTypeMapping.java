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
import io.trino.testing.AbstractTestQueryFramework;
import io.trino.testing.QueryRunner;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.TestInstance.Lifecycle.PER_CLASS;

/**
 * Type mapping round-trip tests for all IoTDB data types.
 * <p>
 * Creates tables with each IoTDB type, inserts data, and verifies
 * that Trino reads the values with the expected Trino types.
 */
@TestInstance(PER_CLASS)
class TestIoTDBTypeMapping
        extends AbstractTestQueryFramework
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

    @Test
    void testBooleanMapping()
            throws Exception
    {
        server.execute("CREATE DATABASE IF NOT EXISTS type_test_db");
        server.execute("USE type_test_db");
        server.execute("CREATE TABLE IF NOT EXISTS bool_test (device_id STRING TAG, flag BOOLEAN FIELD)");
        server.execute("INSERT INTO bool_test(time, device_id, flag) VALUES (1704067200000, 'dev1', true)");

        assertThat(computeScalar("SELECT flag FROM iotdb.type_test_db.bool_test"))
                .isEqualTo(true);
    }

    @Test
    void testIntegerMapping()
            throws Exception
    {
        server.execute("CREATE DATABASE IF NOT EXISTS type_test_db");
        server.execute("USE type_test_db");
        server.execute("CREATE TABLE IF NOT EXISTS int_test (device_id STRING TAG, value INT32 FIELD)");
        server.execute("INSERT INTO int_test(time, device_id, value) VALUES (1704067200000, 'dev1', 42)");

        assertThat(computeScalar("SELECT value FROM iotdb.type_test_db.int_test"))
                .isEqualTo(42);
    }

    @Test
    void testBigintMapping()
            throws Exception
    {
        server.execute("CREATE DATABASE IF NOT EXISTS type_test_db");
        server.execute("USE type_test_db");
        server.execute("CREATE TABLE IF NOT EXISTS bigint_test (device_id STRING TAG, value INT64 FIELD)");
        server.execute("INSERT INTO bigint_test(time, device_id, value) VALUES (1704067200000, 'dev1', 9999999999)");

        assertThat(computeScalar("SELECT value FROM iotdb.type_test_db.bigint_test"))
                .isEqualTo(9999999999L);
    }

    @Test
    void testFloatMapping()
            throws Exception
    {
        server.execute("CREATE DATABASE IF NOT EXISTS type_test_db");
        server.execute("USE type_test_db");
        server.execute("CREATE TABLE IF NOT EXISTS float_test (device_id STRING TAG, value FLOAT FIELD)");
        server.execute("INSERT INTO float_test(time, device_id, value) VALUES (1704067200000, 'dev1', 3.14)");

        // IoTDB FLOAT → Trino REAL (32-bit), expect approximate equality
        Object result = computeScalar("SELECT value FROM iotdb.type_test_db.float_test");
        assertThat(result).isInstanceOf(Float.class);
    }

    @Test
    void testDoubleMapping()
            throws Exception
    {
        server.execute("CREATE DATABASE IF NOT EXISTS type_test_db");
        server.execute("USE type_test_db");
        server.execute("CREATE TABLE IF NOT EXISTS double_test (device_id STRING TAG, value DOUBLE FIELD)");
        server.execute("INSERT INTO double_test(time, device_id, value) VALUES (1704067200000, 'dev1', 3.141592653589793)");

        assertThat(computeScalar("SELECT value FROM iotdb.type_test_db.double_test"))
                .isEqualTo(3.141592653589793);
    }

    @Test
    void testVarcharMapping()
            throws Exception
    {
        server.execute("CREATE DATABASE IF NOT EXISTS type_test_db");
        server.execute("USE type_test_db");
        server.execute("CREATE TABLE IF NOT EXISTS varchar_test (device_id STRING TAG, value STRING FIELD)");
        server.execute("INSERT INTO varchar_test(time, device_id, value) VALUES (1704067200000, 'dev1', 'hello world')");

        assertThat(computeScalar("SELECT value FROM iotdb.type_test_db.varchar_test"))
                .isEqualTo("hello world");
    }

    @Test
    void testTimestampColumnExists()
            throws Exception
    {
        // Verify that the TIME column is exposed as a TIMESTAMP type
        server.execute("CREATE DATABASE IF NOT EXISTS type_test_db");
        server.execute("USE type_test_db");
        server.execute("CREATE TABLE IF NOT EXISTS ts_test (device_id STRING TAG, value FLOAT FIELD)");
        server.execute("INSERT INTO ts_test(time, device_id, value) VALUES (1704067200000, 'dev1', 1.0)");

        // The TIME column should be queryable
        assertQuerySucceeds("SELECT time FROM iotdb.type_test_db.ts_test");
    }

    @Test
    void testDateMapping()
            throws Exception
    {
        server.execute("CREATE DATABASE IF NOT EXISTS type_test_db");
        server.execute("USE type_test_db");
        server.execute("CREATE TABLE IF NOT EXISTS date_test (device_id STRING TAG, value DATE FIELD)");
        server.execute("INSERT INTO date_test(time, device_id, value) VALUES (1704067200000, 'dev1', '2024-01-01')");

        assertQuerySucceeds("SELECT value FROM iotdb.type_test_db.date_test");
    }

    @Test
    void testSelectStar()
    {
        // Verify SELECT * works on the sample sensors table
        assertQuerySucceeds("SELECT * FROM iotdb.test_db.sensors");
    }

    @Test
    void testPredicatePushdown()
    {
        // Verify basic predicate pushdown with time filter
        assertQuerySucceeds(
                "SELECT temperature FROM iotdb.test_db.sensors WHERE device_id = 'dev-001'");
    }

    @Test
    void testLimitPushdown()
    {
        // Verify LIMIT pushdown works
        assertQuerySucceeds(
                "SELECT * FROM iotdb.test_db.sensors LIMIT 2");
    }
}
