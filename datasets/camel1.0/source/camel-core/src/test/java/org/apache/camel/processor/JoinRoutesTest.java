/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.processor;

import org.apache.camel.ContextTestSupport;
import org.apache.camel.Endpoint;
import org.apache.camel.Exchange;
import org.apache.camel.Message;
import org.apache.camel.Processor;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.mock.MockEndpoint;

/**
 * @version $Revision: 1.1 $
 */
public class JoinRoutesTest extends ContextTestSupport {
    protected Endpoint<Exchange> startEndpoint;
    protected MockEndpoint resultEndpoint;

    public void testMessagesThroughDifferentRoutes() throws Exception {
        resultEndpoint.expectedBodiesReceived("one", "two", "three");

        sendMessage("bar", "one");
        sendMessage("cheese", "two");
        sendMessage("somethingUndefined", "three");

        resultEndpoint.assertIsSatisfied();
    }

    protected void sendMessage(final Object headerValue, final Object body) throws Exception {
        template.send(startEndpoint, new Processor() {
            public void process(Exchange exchange) {
                // now lets fire in a message
                Message in = exchange.getIn();
                in.setBody(body);
                in.setHeader("foo", headerValue);
            }
        });
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        startEndpoint = resolveMandatoryEndpoint("direct:a");
        resultEndpoint = (MockEndpoint) resolveMandatoryEndpoint("mock:result");
    }

    protected RouteBuilder createRouteBuilder() {
        return new RouteBuilder() {
            public void configure() {
                from("direct:a").choice()
                        .when(header("foo").isEqualTo("bar")).to("direct:b")
                        .when(header("foo").isEqualTo("cheese")).to("direct:c")
                        .otherwise().to("direct:d");

                from("direct:b").to("mock:result");
                from("direct:c").to("mock:result");
                from("direct:d").to("mock:result");
            }
        };
    }
}
