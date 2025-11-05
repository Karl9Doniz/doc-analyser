import json
import os
from typing import Any, Dict, List, Optional
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
        self.tool_history = {}
        self.tool_attempts = {}
        self.successful_tools = set()
        self.tool_execution_order = []
        self.chart_region_attempts = {}
        self.observations = []
        self.decisions = []
        self.final_answer = None

    def add_tool_result(self, tool_name: str, result: Dict[str, Any]):
        """Store tool execution result"""
        self.tool_results[f"{tool_name}_{self.step}"] = result
        self.tool_history.setdefault(tool_name, []).append(result)
        self.tool_attempts[tool_name] = self.tool_attempts.get(tool_name, 0) + 1
        if result.get("success"):
            self.successful_tools.add(tool_name)
        self.tool_execution_order.append(tool_name)
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
            "load_image",
            "preprocess",
            "extract_text",
            "score_document",
            "analyze_layout",
            "detect_regions",
            "detect_formulas",
            "recognize_formulas",
            "extract_chart_data",
            "generate_output",
        ]
        self.max_tool_retries = 2
        self.visual_keywords = {
            "figure",
            "fig.",
            "fig ",
            "chart",
            "graph",
            "diagram",
            "plot",
            "image",
            "visual",
            "table",
            "trend",
            "number",
            "value",
        }
        self.formula_keywords = {
            "equation",
            "equations",
            "formula",
            "formulas",
            "mathematics",
            "math",
            "derivation",
            "latex",
            "expression",
            "proof",
            "symbol",
        }
        self.chart_model_path = os.getenv("MINICPM_MODEL_PATH")
        self.chart_mmproj_path = os.getenv("MINICPM_MMPROJ_PATH")

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
            mandatory_tool = self._next_mandatory_tool(state)
            if mandatory_tool:
                next_action = {
                    "action": "TOOL_CALL",
                    "tool": mandatory_tool,
                    "params": {},
                    "reasoning": f"Enforcing mandatory tool '{mandatory_tool}' to satisfy pipeline prerequisites."
                }
                print(f"\033[94mGuard: auto-calling {mandatory_tool} to satisfy prerequisites.\033[0m")
            else:
                chart_action = self._next_chart_action(state)
                if chart_action:
                    region_label = chart_action["params"].get("region_id", "unknown")
                    print(f"\033[94mGuard: auto-calling extract_chart_data for region {region_label}.\033[0m")
                    self._execute_tool_call(state, chart_action)
                    reflection_count = 0
                    continue

                next_action = self._decide_next_action(state)
                if next_action["action"] == "FINAL_ANSWER":
                    pending_final = self._next_mandatory_tool(state, require_completion=True)
                    if pending_final:
                        next_action = {
                            "action": "TOOL_CALL",
                            "tool": pending_final,
                            "params": {},
                            "reasoning": f"Mandatory tool '{pending_final}' still pending before final answer."
                        }
                        print(f"\033[94mGuard: blocking FINAL_ANSWER until {pending_final} completes.\033[0m")

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
        If they want visual elements (figures/tables/charts), include analyze_layout (Docling) before detect_regions as the fallback, and mention extract_chart_data for reading figure content when numeric insight is required.
        If they reference equations or formulas, add detect_formulas -> recognize_formulas after the core text steps so you can deliver LaTeX output.

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
            "scientific": "After basic text analysis, call analyze_layout to capture figures/tables precisely; run detect_formulas followed by recognize_formulas whenever equations are referenced; use extract_chart_data on figure crops when quantitative insight is requested. Fall back to detect_regions only if needed.",
            "business": "Emphasize analyze_layout for charts/tables, then use extract_chart_data to recover numeric values before summarising financial metrics.",
            "legal": "Focus on load/preprocess/extract_text/score first; use analyze_layout only if visual exhibits are referenced, and chart extraction only when necessary.",
            "general": "Use standard analysis workflow; rely on analyze_layout for figures/tables, extract_chart_data for chart comprehension, and detect_regions only as a backup."
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
        - For charts/tables/figures, call analyze_layout (Docling) first; use detect_regions only if analyze_layout is unavailable
        - When figures are present and the user asks for trends, numbers, or chart insights, call extract_chart_data with the matching region
        - When equations or formulas are likely, run detect_formulas then recognize_formulas to capture LaTeX before final reasoning
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
        original_params = dict(params)

        # Add image_path to params if needed and clean up params
        image_tools = {
            "load_image",
            "preprocess",
            "extract_text",
            "score_document",
            "detect_regions",
            "analyze_layout",
            "extract_chart_data",
            "detect_formulas",
            "recognize_formulas",
        }
        if tool_name in image_tools:
            if tool_name in {
                "analyze_layout",
                "detect_regions",
                "extract_chart_data",
                "detect_formulas",
                "recognize_formulas",
            }:
                params["image_path"] = state.image_path
            elif "image_path" not in params:
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
            elif tool_name == "analyze_layout":
                valid_params.update({"include_figures", "include_tables"})
            elif tool_name == "detect_regions":
                valid_params.update({"detect_figures", "detect_tables", "detect_captions"})
            elif tool_name == "detect_formulas":
                valid_params.update({"min_confidence", "max_regions", "padding"})
            elif tool_name == "recognize_formulas":
                valid_params.update({"regions", "max_regions", "padding"})
            elif tool_name == "extract_chart_data":
                if "question" not in params:
                    params["question"] = self._build_chart_question(state.user_request)
                if "model_path" not in params and self.chart_model_path:
                    params["model_path"] = self.chart_model_path
                if "mmproj_path" not in params and self.chart_mmproj_path:
                    params["mmproj_path"] = self.chart_mmproj_path
                valid_params.update(
                    {
                        "bbox",
                        "normalized_bbox",
                        "page_dimensions",
                        "question",
                        "region_id",
                        "model_path",
                        "mmproj_path",
                        "max_tokens",
                        "temperature",
                    }
                )
            elif tool_name == "generate_output":
                valid_params.update({"output_dir", "text", "metadata"})

            params = {k: v for k, v in params.items() if k in valid_params}

        # Special handling for score_document - needs text_data from extract_text
        if tool_name == "score_document" and "text_data" not in params:
            text_result = self._get_latest_tool_result("extract_text", state)
            if text_result and text_result.get("success"):
                params["text_data"] = text_result["data"]
        elif tool_name == "recognize_formulas" and "regions" not in params:
            formula_result = self._get_latest_tool_result("detect_formulas", state)
            if formula_result.get("success"):
                detected_regions = formula_result.get("data", {}).get("regions")
                if detected_regions:
                    params["regions"] = detected_regions

        print(f"Executing {tool_name} with {params}")
        result = execute_tool(tool_name, **params)
        state.add_tool_result(tool_name, result)

        if tool_name == "extract_chart_data":
            region_key = original_params.get("region_id")
            if region_key is not None:
                key_str = str(region_key)
                state.chart_region_attempts[key_str] = state.chart_region_attempts.get(key_str, 0) + 1

        if result["success"]:
            print(f"\033[92m{tool_name} completed\033[0m")
            self._maybe_visualize_regions(state, tool_name, result, params.get("image_path", state.image_path))
        else:
            print(f"\033[91m{tool_name} failed: {result['error']}\033[0m")

    def _get_latest_tool_result(self, tool_name: str, state: AgentState) -> Dict[str, Any]:
        """Get the most recent result from a specific tool"""
        matching_results = [
            result for key, result in state.tool_results.items()
            if key.startswith(tool_name)
        ]
        if matching_results:
            return matching_results[-1]
        history = state.tool_history.get(tool_name)
        return history[-1] if history else {}

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
        detect_formula_result = self._get_latest_tool_result("detect_formulas", state)
        recognize_formula_result = self._get_latest_tool_result("recognize_formulas", state)

        text = text_result.get("data", {}).get("text", "") if text_result.get("success") else ""
        score_data = score_result.get("data", {}) if score_result.get("success") else {}

        formula_context = ""
        if recognize_formula_result.get("success"):
            recognized = recognize_formula_result.get("data", {}).get("formulas", []) or []
            detected_regions: Dict[int, Dict[str, Any]] = {}
            if detect_formula_result.get("success"):
                detected_regions = {
                    region.get("region_id"): region
                    for region in (detect_formula_result.get("data", {}).get("regions") or [])
                    if region.get("region_id") is not None
                }

            formula_lines: List[str] = []
            for entry in recognized:
                latex = (entry.get("latex") or "").strip()
                if not latex:
                    continue
                region_id = entry.get("region_id")
                region_meta = detected_regions.get(region_id, {})
                snippet = region_meta.get("text_hint") or ""
                snippet = " ".join(snippet.replace("\n", " ").split())
                source = region_meta.get("source") or entry.get("source") or "unknown"
                confidence = entry.get("confidence")
                conf_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
                parts = [f"LaTeX: {latex}", f"source: {source}", f"confidence: {conf_text}"]
                if snippet:
                    parts.append(f"context: {snippet}")
                formula_lines.append("- " + "; ".join(parts))

            if formula_lines:
                formula_context = "FORMULA EXTRACTION SUMMARY:\n" + "\n".join(formula_lines)

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

        if formula_context:
            context_block = f"{context}\n\n{formula_context}"
        else:
            context_block = context

        prompt = f"""{context_block}

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

3. FORMULA ANALYSIS:
   - Catalogue every extracted equation with its LaTeX transcription
   - Explain variables, constants, and the relationship expressed
   - Highlight assumptions, approximations, or missing context

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

    def _build_chart_question(self, user_request: str) -> str:
        return (
            "You are an assistant that extracts structured data from charts. "
            "Return a JSON object with keys: title, axes, legend, series, annotations, summary. "
            "Each axis entry should include label, units, and range (if visible). "
            "Each series entry should have a name and a list of points with exact numeric values. "
            "Summarize the main trend using the extracted numbers. "
            f"The user request is: {user_request}"
        )

    def _figure_regions(self, state: AgentState) -> List[Dict[str, Any]]:
        regions: List[Dict[str, Any]] = []
        for tool_name in ("analyze_layout", "detect_regions"):
            history = state.tool_history.get(tool_name, [])
            for entry in reversed(history):
                if not entry.get("success"):
                    continue
                data = entry.get("data") or {}
                page_dimensions = data.get("page_dimensions", {})
                raw_regions = data.get("regions", [])
                for idx, region in enumerate(raw_regions):
                    if region.get("type") != "figure":
                        continue
                    raw_id = region.get("region_id", idx)
                    unique_id = f"{tool_name}:{raw_id}"
                    page = region.get("page")
                    dimensions = None
                    if isinstance(page_dimensions, dict) and page_dimensions:
                        if page in page_dimensions:
                            dimensions = page_dimensions[page]
                        elif page is not None and str(page) in page_dimensions:
                            dimensions = page_dimensions[str(page)]
                        else:
                            dimensions = next(iter(page_dimensions.values()))
                    regions.append(
                        {
                            "id": str(unique_id),
                            "source": tool_name,
                            "bbox": region.get("bbox"),
                            "normalized_bbox": region.get("normalized_bbox"),
                            "page": page,
                            "page_dimensions": dimensions,
                            "confidence": region.get("confidence"),
                        }
                    )
                break  # use the most recent successful result per tool
        return regions

    def _chart_regions_pending(self, state: AgentState) -> List[Dict[str, Any]]:
        figures = self._figure_regions(state)
        if not figures:
            return []

        processed = set()
        for entry in state.tool_history.get("extract_chart_data", []):
            if not entry.get("success"):
                continue
            data = entry.get("data") or {}
            region_id = data.get("region_id")
            if region_id is not None:
                processed.add(str(region_id))

        pending: List[Dict[str, Any]] = []
        for region in figures:
            region_id = region["id"]
            attempts = state.chart_region_attempts.get(region_id, 0)
            if region_id in processed or attempts >= self.max_tool_retries:
                continue
            if not region.get("bbox") and not region.get("normalized_bbox"):
                continue
            pending.append(region)
        return pending

    def _should_attempt_chart_vqa(self, state: AgentState) -> bool:
        request_lower = state.user_request.lower()
        return any(keyword in request_lower for keyword in self.visual_keywords)

    def _next_chart_action(self, state: AgentState) -> Optional[Dict[str, Any]]:
        if not self._should_attempt_chart_vqa(state):
            return None

        for region in self._chart_regions_pending(state):
            params = {
                "image_path": state.image_path,
                "bbox": region.get("bbox"),
                "normalized_bbox": region.get("normalized_bbox"),
                "page_dimensions": region.get("page_dimensions"),
                "region_id": region["id"],
                "question": self._build_chart_question(state.user_request),
                "model_path": self.chart_model_path,
                "mmproj_path": self.chart_mmproj_path,
            }
            params = {k: v for k, v in params.items() if v is not None}
            return {
                "action": "TOOL_CALL",
                "tool": "extract_chart_data",
                "params": params,
                "reasoning": f"Extract chart data for region {region['id']} to address the user request.",
            }
        return None

    def _mandatory_tool_sequence(self, state: AgentState) -> list[str]:
        sequence = ["load_image", "preprocess", "extract_text"]
        if self._needs_visual_analysis(state):
            sequence.extend(["analyze_layout", "detect_regions"])
        if self._needs_formula_analysis(state):
            sequence.extend(["detect_formulas", "recognize_formulas"])
        return sequence

    def _needs_visual_analysis(self, state: AgentState) -> bool:
        request_lower = state.user_request.lower()
        return any(keyword in request_lower for keyword in self.visual_keywords)

    def _needs_formula_analysis(self, state: AgentState) -> bool:
        request_lower = state.user_request.lower()
        if any(keyword in request_lower for keyword in self.formula_keywords):
            return True

        text_result = self._get_latest_tool_result("extract_text", state)
        if text_result.get("success"):
            text = text_result.get("data", {}).get("text", "") or ""
            text_lower = text.lower()
            formula_triggers = [
                "\\begin{equation}",
                "\\begin{align}",
                "\\frac",
                "$",
                " equation ",
                " formula ",
                " theorem ",
            ]
            if any(trigger in text_lower for trigger in formula_triggers):
                return True
        return False

    def _tool_requirement_met(self, state: AgentState, tool: str) -> bool:
        if tool == "analyze_layout":
            if self._tool_has_regions(state, "analyze_layout"):
                return True
            return self._tool_has_regions(state, "detect_regions")
        if tool == "detect_regions":
            if self._tool_has_regions(state, "detect_regions"):
                return True
            return self._tool_has_regions(state, "analyze_layout")
        return tool in state.successful_tools

    def _maybe_visualize_regions(
        self,
        state: AgentState,
        tool_name: str,
        result: Dict[str, Any],
        image_path: Optional[str],
    ) -> None:
        if tool_name not in {"analyze_layout", "detect_formulas"}:
            return
        if not result.get("success"):
            return

        data = result.get("data", {})
        regions = data.get("regions")
        if not regions:
            return

        image_to_use = image_path or state.image_path
        base_name = os.path.splitext(os.path.basename(state.image_path))[0]
        suffix = "docling" if tool_name == "analyze_layout" else "formulas"
        output_dir = os.path.join("out", "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}_regions.png")

        vis_result = execute_tool(
            "visualize_regions",
            image_path=image_to_use,
            regions=regions,
            output_path=output_path,
            show_labels=True,
        )
        key = f"visualize_regions_{suffix}_{state.step}"
        state.tool_results[key] = vis_result
        state.tool_history.setdefault("visualize_regions", []).append(vis_result)
        if vis_result.get("success"):
            state.successful_tools.add("visualize_regions")
        observation = f"Rendered {len(regions)} {suffix} regions to {output_path}"
        state.observations.append(observation)
        print(f"\033[96m{observation}\033[0m")

    def _next_mandatory_tool(self, state: AgentState, require_completion: bool = False) -> Optional[str]:
        sequence = self._mandatory_tool_sequence(state)
        for tool in sequence:
            if tool == "detect_regions" and self._tool_requirement_met(state, "analyze_layout"):
                continue
            if self._tool_requirement_met(state, tool):
                continue

            attempts = state.tool_attempts.get(tool, 0)

            if tool == "analyze_layout" and attempts > 0 and not self._tool_has_regions(state, "analyze_layout"):
                # Docling already tried without yielding regions; rely on detect_regions fallback
                continue

            if tool == "detect_regions" and attempts > 0 and not self._tool_has_regions(state, "detect_regions"):
                # Detect regions already attempted without new regions; allow flow to continue
                continue

            if not require_completion and attempts >= self.max_tool_retries:
                continue

            return tool
        return None

    def _tool_has_regions(self, state: AgentState, tool_name: str) -> bool:
        history = state.tool_history.get(tool_name, [])
        for result in reversed(history):
            if not result.get("success"):
                continue
            data = result.get("data")
            if not isinstance(data, dict):
                continue
            regions = data.get("regions")
            if regions:
                return True
        return False
