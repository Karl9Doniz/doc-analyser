import json
import os
from typing import Dict, Any
from openai import OpenAI
try:
    from .tools import execute_tool
except ImportError:
    from tools import execute_tool


class AgentState:
    """Manages agent execution state"""
    def __init__(self, user_request: str, image_path: str):
        self.user_request = user_request
        self.image_path = image_path
        self.step = 0
        self.tool_results = {}
        self.observations = []
        self.decisions = []
        self.final_answer = None

    def add_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Store tool execution result"""
        self.tool_results[f"{tool_name}_{self.step}"] = result
        self.step += 1

    def get_context_for_llm(self) -> str:
        """Format current state for LLM"""
        context = f"""USER REQUEST: {self.user_request}
        IMAGE PATH: {self.image_path}
        CURRENT STEP: {self.step}

        TOOL RESULTS SO FAR:
        """
        for tool_id, result in self.tool_results.items():
            context += f"\n{tool_id}: {json.dumps(result, indent=2)}\n"

        return context


class DocumentOrchestrator:

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.available_tools = [
            "load_image", "preprocess", "extract_text",
            "score_document", "detect_regions", "generate_output"
        ]

    def analyze(self, user_request: str, image_path: str, output_dir: str = "out") -> Dict[str, Any]:
        state = AgentState(user_request, image_path)

        # Initial planning
        plan = self._get_initial_plan(state)
        print(f"Plan: {plan}")

        # Execute tools based on LLM decisions
        max_steps = 10
        reflection_count = 0
        max_reflections = 2  # Prevent infinite reflection loops

        while state.step < max_steps and not state.final_answer:
            next_action = self._decide_next_action(state)

            if next_action["action"] == "TOOL_CALL":
                self._execute_tool_call(state, next_action)
                reflection_count = 0
            elif next_action["action"] == "REFLECT":
                reflection_count += 1
                if reflection_count > max_reflections:
                    print("\033[93mToo many reflections - proceeding to final answer\033[0m")
                    state.final_answer = self._generate_final_report(state, output_dir)
                    break
                reflection = self._reflect_on_progress(state)
                state.decisions.append(f"Step {state.step}: {reflection}")
            elif next_action["action"] == "FINAL_ANSWER":
                state.final_answer = self._generate_final_report(state, output_dir)
                break
            else:
                print(f"\033[91mUnknown action: {next_action}\033[0m")
                break

        return state.final_answer or {"error": "Analysis incomplete", "state": state.__dict__}

    def _get_initial_plan(self, state: AgentState) -> str:
        """Get LLM's initial analysis plan"""
        prompt = f"""You are a document analysis agent. Given this request:
        "{state.user_request}"

        For image: {state.image_path}

        Available tools: {', '.join(self.available_tools)}

        Create a brief 2-3 step plan. Focus on the user's specific needs.
        If they just want to know "is this a document?", you need: load_image -> preprocess -> extract_text -> score_document
        If they want specific data extraction, add those steps.

        Return just the plan in 1-2 sentences."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def _decide_next_action(self, state: AgentState) -> Dict[str, Any]:
        context = state.get_context_for_llm()

        # Detect document domain for smart tool selection
        domain = self._detect_document_domain(state.user_request, context)

        domain_guidance = {
            "scientific": "For scientific documents, prioritize figure detection and table extraction after basic text analysis.",
            "business": "For business documents, focus on extracting financial data, charts, and performance metrics.",
            "legal": "For legal documents, emphasize text extraction and table analysis for terms, dates, and obligations.",
            "general": "Use standard analysis workflow with attention to any visual elements."
        }

        prompt = f"""{context}

        You are orchestrating document analysis for a {domain} document. Based on the current state, decide the next action:

        AVAILABLE ACTIONS:
        1. TOOL_CALL - Call one of: {', '.join(self.available_tools)}
        2. REFLECT - Think about progress and adjust strategy
        3. FINAL_ANSWER - Generate final report (only when analysis is complete)

        DECISION RULES:
        - Start with load_image if not done
        - Always preprocess before OCR
        - Score document after getting text
        - For complex documents with tables/charts, use detect_regions tool
        - Call FINAL_ANSWER when you have enough info to answer the user's request

        DOMAIN GUIDANCE: {domain_guidance.get(domain, domain_guidance["general"])}

        Return JSON: {{"action": "TOOL_CALL|REFLECT|FINAL_ANSWER", "tool": "tool_name", "params": {{}}, "reasoning": "why"}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )

        try:
            decision = json.loads(response.choices[0].message.content.strip())
            print(f"Decision: {decision['action']} - {decision.get('reasoning', '')}")
            return decision
        except json.JSONDecodeError:
            return {"action": "FINAL_ANSWER", "reasoning": "Failed to parse decision"}

    def _execute_tool_call(self, state: AgentState, action: Dict[str, Any]):
        """Execute the requested tool"""
        tool_name = action.get("tool")
        params = action.get("params", {})

        # Add image_path to params if needed and clean up params
        if tool_name in ["load_image", "preprocess", "extract_text", "score_document", "detect_regions"]:
            if "image_path" not in params:
                # Use preprocessed image if available, otherwise original
                preprocessed_result = self._get_latest_tool_result("preprocess", state)
                if preprocessed_result and preprocessed_result.get("success"):
                    params["image_path"] = preprocessed_result["data"]["output_path"]
                else:
                    params["image_path"] = state.image_path

            # Clean up any extra parameters that tools don't expect
            valid_params = {"image_path"}
            if tool_name == "preprocess":
                valid_params.add("method")
            elif tool_name == "extract_text":
                valid_params.add("config")
            elif tool_name == "score_document":
                valid_params.add("text_data")
            elif tool_name == "detect_regions":
                valid_params.update({"detect_figures", "detect_tables", "detect_captions"})
            elif tool_name == "generate_output":
                valid_params.update({"output_dir", "text", "metadata"})

            params = {k: v for k, v in params.items() if k in valid_params}

        # Special handling for score_document - needs text_data from extract_text
        if tool_name == "score_document" and "text_data" not in params:
            text_result = self._get_latest_tool_result("extract_text", state)
            if text_result and text_result.get("success"):
                params["text_data"] = text_result["data"]

        print(f"Executing {tool_name} with {params}")
        result = execute_tool(tool_name, **params)
        state.add_tool_result(tool_name, result)

        if result["success"]:
            print(f"\033[92m{tool_name} completed\033[0m")
        else:
            print(f"\033[91m{tool_name} failed: {result['error']}\033[0m")

    def _get_latest_tool_result(self, tool_name: str, state: AgentState) -> Dict[str, Any]:
        """Get the most recent result from a specific tool"""
        matching_results = [
            result for key, result in state.tool_results.items()
            if key.startswith(tool_name)
        ]
        return matching_results[-1] if matching_results else {}

    def _reflect_on_progress(self, state: AgentState) -> str:
        """LLM reflects on current progress"""
        context = state.get_context_for_llm()

        prompt = f"""{context}

Reflect on the current analysis progress. What have we learned? What should we do differently?
Keep it brief (1-2 sentences)."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )

        reflection = response.choices[0].message.content.strip()
        print(f"Reflection: {reflection}")
        return reflection

    def _generate_final_report(self, state: AgentState, output_dir: str) -> Dict[str, Any]:
        """LLM generates final human-readable report"""
        context = state.get_context_for_llm()

        # Get key results
        text_result = self._get_latest_tool_result("extract_text", state)
        score_result = self._get_latest_tool_result("score_document", state)

        text = text_result.get("data", {}).get("text", "") if text_result.get("success") else ""
        score_data = score_result.get("data", {}) if score_result.get("success") else {}

        # Generate output files if we have text
        if text and score_data.get("final_score", 0) > 0.6:
            output_result = execute_tool("generate_output",
                                       output_dir=output_dir,
                                       text=text,
                                       metadata={
                                           "is_document": True,
                                           "confidence": score_data.get("final_score", 0)
                                       })

        # Enhanced domain-adaptive analysis
        analysis_guidance = self._create_domain_adaptive_analysis(state.user_request, context)

        prompt = f"""{context}

        {analysis_guidance}

        Generate a final report answering the user's request: "{state.user_request}"

        Based on the tool results, provide:
        1. Clear answer to their question - if specific data is not found, say so clearly
        2. Confidence level and reasoning
        3. Key findings from what IS available
        4. Any recommendations

        IMPORTANT: If the user asked for specific data (dates, names, amounts) that you cannot find in the extracted text, clearly state "The requested information was not found in the document" or "The document contains placeholders but not the actual values."

        Be conversational and helpful. Return JSON with: {{"answer": "main response", "confidence": 0.0-1.0, "summary": "brief summary", "recommendations": []}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )

        try:
            report = json.loads(response.choices[0].message.content.strip())

            # Add technical details
            report["technical_details"] = {
                "steps_taken": state.step,
                "tools_used": list(set(key.split('_')[0] for key in state.tool_results.keys())),
                "all_results": state.tool_results
            }

            return report
        except json.JSONDecodeError:
            return {
                "answer": response.choices[0].message.content.strip(),
                "confidence": score_data.get("final_score", 0),
                "technical_details": {"raw_response": True}
            }

    def _create_domain_adaptive_analysis(self, user_request: str, context: str) -> str:
        """Create domain-specific analysis guidance using Charts-of-Thought approach"""
        domain = self._detect_document_domain(user_request, context)

        base_analysis = """
ENHANCED DOCUMENT ANALYSIS FRAMEWORK:

Use this structured approach for comprehensive document understanding:

1. DOCUMENT OVERVIEW:
   - Identify all visual elements: tables, charts, graphs, diagrams, images
   - Note document structure: headers, sections, layouts
   - Assess text quality and any unclear areas

2. VISUAL ELEMENT ANALYSIS:
   For each table/chart/graph found:
   - Type: (data table, line graph, bar chart, pie chart, flowchart, etc.)
   - Title/Caption: Extract exact text
   - Headers/Labels: Column names, axis labels, legend items
   - Data Values: Key numbers, percentages, dates, amounts
   - Relationships: Trends, comparisons, correlations shown
   - Quality Note: Mark any values that are unclear or unreadable

3. TEXTUAL CONTENT ANALYSIS:
   - Key facts, findings, conclusions
   - Names, dates, locations, amounts mentioned
   - Relationships between text and visual elements
   - Cross-references between sections
"""

        # Add domain-specific guidance
        domain_guidance = {
            "scientific": """
4. SCIENTIFIC DOCUMENT SPECIFICS:
   - Research objectives and hypotheses
   - Methodology and experimental setup
   - Statistical significance and confidence intervals
   - Control groups and variables
   - Conclusions and limitations stated

5. DATA INTERPRETATION:
   - What do the graphs/tables prove or disprove?
   - Are there error bars, confidence intervals, or uncertainty measures?
   - How do results compare to previous studies or benchmarks?
""",
            "business": """
4. BUSINESS DOCUMENT SPECIFICS:
   - Financial metrics and KPIs
   - Performance trends and projections
   - Market data and competitive analysis
   - Risk factors and opportunities
   - Strategic recommendations

5. BUSINESS INTERPRETATION:
   - What do the numbers tell us about performance?
   - Are targets being met or missed?
   - What are the key business drivers shown?
""",
            "legal": """
4. LEGAL DOCUMENT SPECIFICS:
   - Parties involved and their roles
   - Legal obligations, rights, and restrictions
   - Important dates, deadlines, and timeframes
   - Financial terms, penalties, and compensation
   - Conditions, exceptions, and termination clauses

5. LEGAL INTERPRETATION:
   - What are the key binding obligations?
   - What are the consequences of non-compliance?
   - Are there any ambiguous terms that need clarification?
""",
            "general": """
4. GENERAL DOCUMENT ANALYSIS:
   - Main purpose and intended audience
   - Key information hierarchy and importance
   - Supporting evidence and documentation
   - Actionable items or next steps

5. COMPREHENSIVE INTERPRETATION:
   - What are the most important takeaways?
   - How do different sections support the main message?
   - What questions does this document answer or raise?
"""
        }

        return base_analysis + domain_guidance.get(domain, domain_guidance["general"]) + """

6. PRECISION AND ACCURACY CHECK:
   - Distinguish between what you can read clearly vs. what you're interpreting
   - Note any assumptions or inferences you're making
   - Mark uncertain values with confidence levels
   - Avoid hallucination - if unsure, say so explicitly

Focus on extracting accurate, verifiable information while providing insightful analysis.
"""

    def _detect_document_domain(self, user_request: str, context: str) -> str:
        """Detect the likely domain of the document based on request and content"""
        request_lower = user_request.lower()
        context_lower = context.lower()

        # Scientific indicators
        scientific_terms = ['research', 'experiment', 'hypothesis', 'data', 'statistical', 'study', 'results', 'analysis', 'graph', 'chart', 'figure', 'table', 'correlation', 'significance']
        if any(term in request_lower for term in scientific_terms) or any(term in context_lower for term in scientific_terms[:5]):
            return "scientific"

        # Business indicators
        business_terms = ['revenue', 'profit', 'sales', 'market', 'financial', 'performance', 'kpi', 'roi', 'budget', 'forecast', 'quarterly', 'annual']
        if any(term in request_lower for term in business_terms) or any(term in context_lower for term in business_terms[:5]):
            return "business"

        # Legal indicators
        legal_terms = ['contract', 'agreement', 'legal', 'clause', 'liability', 'obligation', 'terms', 'conditions', 'jurisdiction', 'penalty']
        if any(term in request_lower for term in legal_terms) or any(term in context_lower for term in legal_terms[:5]):
            return "legal"

        return "general"