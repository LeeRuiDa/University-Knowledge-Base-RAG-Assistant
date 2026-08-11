export const EXAMPLE_PROMPTS = [
  {
    label: "Internship credit",
    question: "How many credits of CSCE 495 count as one tech elective course?",
  },
  {
    label: "Registration calendar",
    question: "When is priority registration for Fall Semester 2025?",
  },
  {
    label: "Late payment fee",
    question: "What late payment fee is assessed on delinquent student accounts?",
  },
] as const;

export const BOUNDARY_PROMPT = {
  label: "Test its boundary",
  question: "What is the deadline to appeal a parking ticket at UNL?",
} as const;
