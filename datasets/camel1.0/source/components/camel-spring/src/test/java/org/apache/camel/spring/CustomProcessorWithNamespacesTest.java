/**
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.spring;

import org.apache.camel.TestSupport;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.Message;
import org.apache.camel.Route;
import org.apache.camel.spring.example.MyProcessor;
import org.apache.camel.CamelTemplate;
import org.springframework.context.support.AbstractXmlApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import java.util.List;

/**
 * @version $Revision: 521586 $
 */
public class CustomProcessorWithNamespacesTest extends TestSupport {
    protected String body = "<hello>world!</hello>";
    protected AbstractXmlApplicationContext applicationContext;

    public void testXMLRouteLoading() throws Exception {
        applicationContext = createApplicationContext();

        SpringCamelContext context = (SpringCamelContext) applicationContext.getBean("camel");
        assertValidContext(context);

        // now lets send a message
        CamelTemplate<Exchange> template = new CamelTemplate<Exchange>(context);
        template.send("direct:start", new Processor() {
            public void process(Exchange exchange) {
                Message in = exchange.getIn();
                in.setHeader("name", "James");
                in.setBody(body);
            }
        });

        List list = MyProcessor.getExchanges();
        assertEquals("Should have received a single exchange: " + list, 1, list.size());
    }

    protected void assertValidContext(SpringCamelContext context) {
        assertNotNull("No context found!", context);

        List<Route> routes = context.getRoutes();
        assertNotNull("Should have some routes defined", routes);
        assertEquals("Number of routes defined", 1, routes.size());
        Route route = routes.get(0);
        log.debug("Found route: " + route);
    }

    protected ClassPathXmlApplicationContext createApplicationContext() {
        return new ClassPathXmlApplicationContext("org/apache/camel/spring/routingUsingProcessor.xml");
    }

    @Override
    protected void tearDown() throws Exception {
        super.tearDown();
        if (applicationContext != null) {
            applicationContext.destroy();
        }
    }
}
