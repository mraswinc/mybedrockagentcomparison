import streamlit as st
import boto3
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Page config
st.set_page_config(
    page_title="Bedrock Agent Model Comparison",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = {}

def get_bedrock_client(region_name='us-west-2'):
    """Initialize Bedrock Agent Runtime client"""
    return boto3.client('bedrock-agent-runtime', region_name=region_name)

def invoke_agent(agent_id, agent_alias_id, session_id, prompt, model_name, region_name):
    """Invoke a Bedrock agent and return the response"""
    try:
        client = get_bedrock_client(region_name)
        
        # Invoke agent - only required parameters
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt
        )
        
        # Process the streaming response
        output_text = ""
        completion = response.get('completion')
        
        if completion:
            for event in completion:
                # Handle chunk events
                if 'chunk' in event:
                    chunk_data = event['chunk']
                    if 'bytes' in chunk_data:
                        output_text += chunk_data['bytes'].decode('utf-8')
        
        return {
            'success': True,
            'response': output_text if output_text else "No response received",
            'model': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error messages
        if 'stopSequences' in error_msg:
            error_msg += "\n\n💡 Solution: This model doesn't support stop sequences. You need to update your Bedrock agent configuration:\n" \
                        "1. Go to AWS Console → Bedrock → Agents\n" \
                        "2. Select your agent and edit it\n" \
                        "3. In the model settings, remove any stop sequences from the Inference Configuration\n" \
                        "4. Save and create a new version/alias"
        
        return {
            'success': False,
            'error': error_msg,
            'model': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Main UI
st.title("🤖 Bedrock Agent Model Comparison")
st.markdown("Compare responses from different Bedrock agent configurations")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # AWS Region
    aws_region = st.selectbox(
        "AWS Region",
        ["us-west-2", "us-east-1", "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"],
        index=0
    )
    
    num_models = st.number_input("Number of models to compare", min_value=2, max_value=4, value=2)
    
    models_config = []
    for i in range(num_models):
        with st.expander(f"Model {i+1} Configuration", expanded=True):
            model_name = st.text_input(f"Model Name", value=f"Model {i+1}", key=f"name_{i}")
            agent_id = st.text_input(f"Agent ID", key=f"agent_{i}")
            agent_alias_id = st.text_input(f"Agent Alias ID", value="TSTALIASID", key=f"alias_{i}")
            session_id = st.text_input(f"Session ID", value=f"session-{i}", key=f"session_{i}")
            
            models_config.append({
                'name': model_name,
                'agent_id': agent_id,
                'agent_alias_id': agent_alias_id,
                'session_id': session_id
            })

# Main content area
prompt = st.text_area("Enter your prompt:", height=100, placeholder="Ask a question to compare responses...")

col1, col2 = st.columns([1, 5])
with col1:
    compare_button = st.button("🚀 Compare Models", type="primary")
with col2:
    clear_button = st.button("🗑️ Clear Results")

if clear_button:
    st.session_state.responses = {}
    st.rerun()

if compare_button and prompt:
    if not all(config['agent_id'] for config in models_config):
        st.error("Please configure all Agent IDs before comparing.")
    else:
        st.session_state.responses = {}
        
        with st.spinner("Invoking all agents simultaneously..."):
            # Invoke all agents simultaneously using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_models) as executor:
                # Submit all tasks
                future_to_config = {
                    executor.submit(
                        invoke_agent,
                        config['agent_id'],
                        config['agent_alias_id'],
                        config['session_id'],
                        prompt,
                        config['name'],
                        aws_region
                    ): config for config in models_config
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        st.session_state.responses[config['name']] = result
                    except Exception as e:
                        st.session_state.responses[config['name']] = {
                            'success': False,
                            'error': str(e),
                            'model': config['name'],
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
            
            # Display results in columns
            cols = st.columns(num_models)
            for idx, config in enumerate(models_config):
                with cols[idx]:
                    st.subheader(config['name'])
                    result = st.session_state.responses.get(config['name'])
                    
                    if result:
                        if result['success']:
                            st.success("✅ Response received")
                            st.markdown("**Response:**")
                            st.write(result['response'])
                            st.caption(f"Timestamp: {result['timestamp']}")
                        else:
                            st.error("❌ Error occurred")
                            st.error(result['error'])
                            st.caption(f"Timestamp: {result['timestamp']}")

# Display saved responses
if st.session_state.responses:
    st.divider()
    st.subheader("📊 Comparison Summary")
    
    # Create comparison table
    comparison_data = []
    for model_name, result in st.session_state.responses.items():
        comparison_data.append({
            'Model': model_name,
            'Status': '✅ Success' if result['success'] else '❌ Failed',
            'Response Length': len(result.get('response', '')) if result['success'] else 'N/A',
            'Timestamp': result['timestamp']
        })
    
    st.table(comparison_data)
    
    # Export option
    if st.button("📥 Export Results as JSON"):
        json_data = json.dumps(st.session_state.responses, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"bedrock_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Footer
st.divider()
st.caption("💡 Tip: Configure your AWS credentials using environment variables or AWS CLI before running this app.")
