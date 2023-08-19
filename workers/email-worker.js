// The email worker subscribes to email events and sends email data and metadata to the agent.
// We want this component to be as dumb as possible. 
// Any processing that needs to understand the data is done by the agent

const EventSource = require('eventsource');
const axios = require('axios');
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables from .env file
dotenv.config({ path: path.join(__dirname, '.env') });

// Construct local API endpoint URL from environment variables
const agentApiHost = process.env.APP_HOST || 'localhost';
const agentApiPort = process.env.APP_PORT || '8000';
const agentApiEndpoint = `http://${agentApiHost}:${agentApiPort}/event`;

// Subscribe to pipedream to receive push notification of incoming events
// Pipedream intermediation allows us to centralize red tape linked to authorization and access to cloud services. 
const eventSourceInit = { headers: {"Authorization": "Bearer f9cc4ec4da451a56ba223d86baa5d197", } }
const es = new EventSource("https://api.pipedream.com/sources/dc_6RuzqKp/sse", eventSourceInit);

console.log("Listening to SSE stream at https://api.pipedream.com/sources/dc_6RuzqKp/sse\n");

// Send all events data to the agent
es.onmessage = async event => {
  console.log(event.data);
  try {
    const response = await axios.post(agentApiEndpoint, event.data);
    console.log(`Data sent to Agent API with status code ${response.status}`);
  } catch (error) {
    console.error(`An error occurred while sending data to the Agent API: ${error}`);
  }
}

es.onerror = function() {
  console.error('An error occurred while connecting to the Pipedream endpoint.');
};
