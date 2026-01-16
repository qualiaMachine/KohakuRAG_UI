# Meeting Notes

This document tracks key decisions and discussions from team meetings.

---

## Week 2 - January 16, 2026

### Chris's Review of Bedrock Proposal

Chris reviewed the [Bedrock integration proposal](bedrock-integration-proposal.md) and confirmed:

- The 3-phase flow (planning, retrieval, generation) matches the paper
- The 4-level hierarchy and bottom-up averaging is correct
- Most Bedrock models should be available by default in `us-east-2`

**Key decisions:**

- Start with small/cheap models for prototype, scale up once plumbing works
- Monitor Cost Explorer regularly (note: 24-hour lag on cost data)
- Use existing SQLite + KohakuVault for now, can explore S3 later
- For ensemble: start with N=1 or N=2, scale up later

**Resources shared:**

- [Generative AI Applications with Amazon Bedrock](https://www.coursera.org/learn/generative-ai-applications-amazon-bedrock) (Coursera)

---

## Week 1 - January 13, 2026

### Initial Meeting - Project Kickoff

**Attendees**: Chris, Blaise, Nils

**Task assignments:**

- **Blaise (local branch)**: Get a local chatbot interface working in Streamlit using a small local model (< 1B params)
- **Nils (bedrock branch)**: Plan conversion to AWS Bedrock, create workflow diagrams

**UI requirements discussed:**

- ChatGPT-like interface
- Chat history
- Show references and explanations with answers

**Resources shared:**

- [KohakuRAG paper](https://drive.google.com/file/d/16cuDubYSbolzZyhu8UqXxvg7jLJUycjS/view)
- [KohakuRAG repo](https://github.com/KohakuBlueleaf/KohakuRAG)
- [Streamlit GenAI course](https://www.coursera.org/learn/fast-prototyping-of-genai-apps-with-streamlit)
- [SageMaker/Bedrock intro](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/index.html)
