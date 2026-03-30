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

import com.google.common.collect.ImmutableList;
import com.google.inject.Inject;
import io.trino.plugin.base.session.SessionPropertiesProvider;
import io.trino.spi.connector.ConnectorSession;
import io.trino.spi.session.PropertyMetadata;

import java.util.List;

import static io.trino.spi.session.PropertyMetadata.booleanProperty;
import static java.lang.Boolean.TRUE;

/**
 * Session-level properties for the IoTDB connector.
 * These can be toggled per-query via SET SESSION iotdb.property_name = value.
 */
public class IoTDBSessionProperties
        implements SessionPropertiesProvider
{
    private static final String PUSHDOWN_ENABLED = "pushdown_enabled";

    private final List<PropertyMetadata<?>> properties;

    @Inject
    public IoTDBSessionProperties(IoTDBConfig config)
    {
        properties = ImmutableList.of(
                booleanProperty(
                        PUSHDOWN_ENABLED,
                        "Enable predicate pushdown to IoTDB",
                        config.isPredicatePushdownEnabled(),
                        false));  // not hidden
    }

    @Override
    public List<PropertyMetadata<?>> getSessionProperties()
    {
        return properties;
    }

    public static boolean isPushdownEnabled(ConnectorSession session)
    {
        return session.getProperty(PUSHDOWN_ENABLED, Boolean.class);
    }
}
