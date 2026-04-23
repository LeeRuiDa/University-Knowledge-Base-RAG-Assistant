# Answer Quality Evaluation

## Summary

- Questions: 18
- Supported questions: 14
- Refusal questions: 4
- Faithfulness mean: 0.9821
- Completeness mean: 2.8571
- Citation usefulness mean: 2.9444
- Refusal behavior mean: 3.0
- Supported pass rate: 0.8571
- Correct refusal rate: 1.0

## Failure Modes

- incomplete_answer: 1
- low_faithfulness: 1

## Failed Examples

### How can a student cancel registration before the semester begins without being charged?
- Category: `registration`
- Expected docs: `unl_add_drop_withdraw, unl_academic_calendar`
- Should refuse: `False`
- Faithfulness: `0.75`
- Completeness: `3`
- Citation usefulness: `3`
- Refusal behavior: `None`
- Answer: A student can cancel registration before the semester begins by dropping all classes through MyRED before the last day to drop a class and receive a full refund, or by notifying the Office of University Registrar. This will avoid tuition charges and grade liability [S1].
- Completeness rationale: The assistant answer clearly explains the process for canceling registration before the semester begins without being charged, specifying the use of MyRED before the last drop date to receive a full refund. It also adds the option of notifying the Office of University Registrar and mentions avoiding tuition charges and grade liability, which aligns with the policy. The answer is complete, concise, and actionable.
- Citation note: Citations point to expected documents and directly support the reference answer.
- Top retrieved docs: `unl_add_drop_withdraw, unl_add_drop_withdraw, unl_academic_calendar`

### On what basis can scholarships be awarded at Nebraska?
- Category: `scholarships`
- Expected docs: `unl_scholarships`
- Should refuse: `False`
- Faithfulness: `1.0`
- Completeness: `1`
- Citation usefulness: `2`
- Refusal behavior: `None`
- Answer: Scholarships at Nebraska can be awarded based on residence, background, professional affiliations, and/or field of study [S2]. Additionally, some scholarships like ROTC scholarships are awarded competitively [S2]. Recipients must also meet satisfactory academic progress standards, including maintaining a minimum GPA and completion rate, to remain eligible for many scholarships [S1][S4]. Specific scholarships, such as the New Nebraskan Tuition Scholarship, have additional criteria like nonresident status, full-time enrollment, credit hour completion, and maintaining good academic standing [S5].
- Completeness rationale: The assistant answer provides some bases for scholarship awards at Nebraska, such as residence, background, professional affiliations, and field of study, which are not mentioned in the reference. It also includes details about eligibility criteria and specific scholarships. However, it misses the key bases listed in the reference answer: academics, financial need, leadership, and service. Since these are central to the question and the reference, the answer is only partially complete.
- Citation note: Citations are relevant, but the supporting overlap is only partial.
- Top retrieved docs: `unl_sap_policy, unl_scholarships, unl_scholarships`
