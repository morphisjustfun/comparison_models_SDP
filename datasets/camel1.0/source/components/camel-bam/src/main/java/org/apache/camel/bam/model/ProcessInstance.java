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
package org.apache.camel.bam.model;

import org.apache.camel.bam.rules.ActivityRules;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import javax.persistence.CascadeType;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.ManyToOne;
import javax.persistence.OneToMany;
import java.util.Collection;
import java.util.Date;
import java.util.HashSet;

/**
 * Represents a single business process
 *
 * @version $Revision: $
 */
@Entity
public class ProcessInstance extends TemporalEntity {
    private static final transient Log log = LogFactory.getLog(ProcessInstance.class);
    private ProcessDefinition processDefinition;
    private Collection<ActivityState> activityStates = new HashSet<ActivityState>();
    private String correlationKey;

    public ProcessInstance() {
        setTimeStarted(new Date());
    }

    public String toString() {
        return getClass().getName() + "[id: " + getId() + ", key: " + getCorrelationKey() + "]";
    }

    // This crap is required to work around a bug in hibernate
    @Override
    @Id
    @GeneratedValue
    public Long getId() {
        return super.getId();
    }

    @ManyToOne(fetch = FetchType.LAZY, cascade = {CascadeType.PERSIST})
    public ProcessDefinition getProcessDefinition() {
        return processDefinition;
    }

    public void setProcessDefinition(ProcessDefinition processDefinition) {
        this.processDefinition = processDefinition;
    }

    @OneToMany(mappedBy = "processInstance", fetch = FetchType.LAZY, cascade = {CascadeType.ALL})
    public Collection<ActivityState> getActivityStates() {
        return activityStates;
    }

    public void setActivityStates(Collection<ActivityState> activityStates) {
        this.activityStates = activityStates;
    }

    public String getCorrelationKey() {
        return correlationKey;
    }

    public void setCorrelationKey(String correlationKey) {
        this.correlationKey = correlationKey;
    }

    // Helper methods
    //-------------------------------------------------------------------------

    /**
     * Returns the activity state for the given activity
     *
     * @param activityRules the activity to find the state for
     * @return the activity state or null if no state could be found for the
     *         given activity
     */
    public ActivityState getActivityState(ActivityRules activityRules) {
        for (ActivityState activityState : getActivityStates()) {
            if (activityState.isActivity(activityRules)) {
                return activityState;
            }
        }
        return null;
    }

    public ActivityState getOrCreateActivityState(ActivityRules activityRules) {
        ActivityState state = getActivityState(activityRules);

        if (state == null) {
            state = createActivityState();
            state.setProcessInstance(this);
            state.setActivityDefinition(activityRules.getActivityDefinition());
            // we don't need to do: getTemplate().persist(state);
        }

        return state;
    }

    protected ActivityState createActivityState() {
        return new ActivityState();
    }
}
