import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator import DocumentOrchestrator


def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_llm.py \"<your question>\" <image_path> [output_dir]")
        print("\nExamples:")
        print('  python analyze_llm.py "Is this a document?" images/doc.jpg')
        print('  python analyze_llm.py "Extract the total amount" images/receipt.jpg')
        print('  python analyze_llm.py "What type of document is this?" images/form.pdf')
        sys.exit(1)

    user_request = sys.argv[1]
    image_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "out"

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print("\033[94mLLM Document Analyzer Starting...\033[0m")
    print(f"Request: {user_request}")
    print(f"Image: {image_path}")
    print("=" * 60)

    try:
        orchestrator = DocumentOrchestrator(api_key)
        result = orchestrator.analyze(user_request, image_path, output_dir)

        print("=" * 60)
        print("\033[92mFINAL ANSWER:\033[0m")

        # Handle JSON responses (including markdown-wrapped JSON)
        answer_text = result.get("answer", "No answer generated")
        confidence = result.get("confidence", 0)
        summary = result.get("summary", "")
        recommendations = result.get("recommendations", [])

        if isinstance(answer_text, str) and "```json" in answer_text:
            try:
                json_start = answer_text.find("```json") + 7
                json_end = answer_text.find("```", json_start)
                json_str = answer_text[json_start:json_end].strip()
                answer_data = json.loads(json_str)

                answer_text = answer_data.get("answer", answer_text)
                confidence = answer_data.get("confidence", confidence)
                summary = answer_data.get("summary", summary)
                recommendations = answer_data.get("recommendations", recommendations)
            except (json.JSONDecodeError, ValueError):
                pass
        elif isinstance(answer_text, str) and answer_text.startswith("{"):
            try:
                answer_data = json.loads(answer_text)
                answer_text = answer_data.get("answer", answer_text)
                confidence = answer_data.get("confidence", confidence)
                summary = answer_data.get("summary", summary)
                recommendations = answer_data.get("recommendations", recommendations)
            except json.JSONDecodeError:
                pass

        print(answer_text)

        if confidence >= 0.8:
            conf_color = "\033[92m"
        elif confidence >= 0.5:
            conf_color = "\033[93m"
        else:
            conf_color = "\033[91m"

        print(f"\n{conf_color}Confidence: {confidence:.1%}\033[0m")

        if summary:
            print(f"\nSummary: {summary}")

        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations:
                print(f"  - {rec}")

        # Save files
        os.makedirs(output_dir, exist_ok=True)

        # Save simple text summary
        summary_path = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"DOCUMENT ANALYSIS RESULT\n")
            f.write(f"========================\n\n")
            f.write(f"Question: {user_request}\n")
            f.write(f"Image: {image_path}\n\n")
            f.write(f"Answer: {answer_text}\n\n")
            f.write(f"Confidence: {confidence:.1%}\n\n")
            if summary:
                f.write(f"Summary: {summary}\n\n")
            if recommendations:
                f.write("Recommendations:\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")

        # Save full technical report
        report_path = os.path.join(output_dir, "llm_report.json")
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nSummary saved: {summary_path}")
        print(f"Full report saved: {report_path}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()