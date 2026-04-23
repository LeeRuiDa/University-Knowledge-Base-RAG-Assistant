# Retrieval Hardening Report

## Summary

| Mode | Retrieval Hit | Top-1 Hit | Citation Hit |
| --- | ---: | ---: | ---: |
| Dense | 0.9831 | 0.7797 | 0.9322 |
| Hybrid | 1.0000 | 0.9492 | 0.9831 |

- Fixed failures: 10
- Remaining failures: 3
- Regressions: 0

## Fixed By Hybrid

### Which course do students enroll in for internship credit in computing?
- Expected doc: `unl_cs_internship_credit`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_cs_major_overview, unl_cs_internship_credit, unl_cs_course_catalog, unl_cs_degree_requirements, unl_cs_course_catalog`
- Hybrid docs: `unl_cs_internship_credit, unl_engineering_internships, unl_engineering_college_overview, unl_engineering_college_overview, unl_undergrad_policies_pdf`

### How many credits of CSCE 495 count as one tech elective course?
- Expected doc: `unl_cs_internship_credit`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_cs_degree_requirements, unl_cs_degree_requirements, unl_cs_internship_credit, unl_cs_degree_requirements, unl_cs_degree_requirements`
- Hybrid docs: `unl_cs_internship_credit, unl_cs_degree_requirements, unl_cs_degree_requirements, unl_cs_course_catalog, unl_undergrad_policies_pdf`

### If students need extra credit to reach 120 hours, how many hours of CSCE 495 may they take and how many count as a tech elective?
- Expected doc: `unl_cs_internship_credit`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_cs_degree_requirements, unl_cs_degree_requirements, unl_cs_internship_credit, unl_cs_degree_requirements, unl_cs_degree_requirements`
- Hybrid docs: `unl_cs_internship_credit, unl_cs_degree_requirements, unl_cs_degree_requirements, unl_undergrad_policies_pdf, unl_sap_policy`

### What should an organization do if it wants to work with Senior Design on a project?
- Expected doc: `unl_cs_senior_design`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_cs_major_overview, unl_cs_senior_design, unl_cs_senior_design, unl_cs_senior_design, unl_cs_senior_design`
- Hybrid docs: `unl_cs_senior_design, unl_cs_senior_design, unl_cs_major_overview, unl_cs_degree_requirements, unl_cs_course_catalog`

### When is priority registration for Fall Semester 2025 according to the academic calendar?
- Expected doc: `unl_academic_calendar`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_add_drop_withdraw, unl_academic_calendar, unl_add_drop_withdraw, unl_academic_calendar, unl_undergrad_policies_pdf`
- Hybrid docs: `unl_academic_calendar, unl_academic_calendar, unl_projected_calendars, unl_undergrad_policies_pdf, unl_engineering_college_overview`

### When is open registration for Fall Semester 2025?
- Expected doc: `unl_academic_calendar`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_add_drop_withdraw, unl_academic_calendar, unl_add_drop_withdraw, unl_academic_calendar, unl_academic_calendar`
- Hybrid docs: `unl_academic_calendar, unl_academic_calendar, unl_undergrad_policies_pdf, unl_undergrad_tuition_2025_2026, unl_student_account_policies`

### What registration fee is due each semester a student registers for classes?
- Expected doc: `unl_undergrad_tuition_2025_2026`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_student_account_policies, unl_undergrad_tuition_2025_2026, unl_billing_process, unl_billing_process, unl_add_drop_withdraw`
- Hybrid docs: `unl_undergrad_tuition_2025_2026, unl_academic_calendar, unl_academic_calendar, unl_student_account_policies, unl_add_drop_withdraw`

### What technology fee is charged and what is the semester cap?
- Expected doc: `unl_undergrad_tuition_2025_2026`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_add_drop_withdraw, unl_undergrad_tuition_2025_2026, unl_billing_process, unl_add_drop_withdraw, unl_billing_process`
- Hybrid docs: `unl_undergrad_tuition_2025_2026, unl_undergrad_tuition_2025_2026, unl_add_drop_withdraw, unl_cs_degree_requirements, unl_sap_policy`

### What transaction fee is charged for an online check payment?
- Expected doc: `unl_payment_credits_refunds`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_pay_bill_online, unl_student_account_policies, unl_student_account_policies, unl_payment_credits_refunds, unl_payment_credits_refunds`
- Hybrid docs: `unl_payment_credits_refunds, unl_payment_credits_refunds, unl_pay_bill_online, unl_pay_bill_online, unl_billing_process`

### What convenience fee is charged for domestic credit card payments?
- Expected doc: `unl_payment_credits_refunds`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: passed
- Dense docs: `unl_student_account_policies, unl_pay_bill_online, unl_undergrad_tuition_2025_2026, unl_payment_credits_refunds, unl_payment_credits_refunds`
- Hybrid docs: `unl_payment_credits_refunds, unl_payment_credits_refunds, unl_pay_bill_online, unl_student_account_policies, unl_billing_process`

## Remaining Failures

### Which senior design sequence can computer science students use to satisfy the senior design experience requirement?
- Expected doc: `unl_cs_degree_requirements`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: Expected document was retrieved but ranked below another source.
- Dense docs: `unl_cs_senior_design, unl_cs_degree_requirements, unl_cs_course_catalog, unl_cs_degree_requirements, unl_cs_major_overview`
- Hybrid docs: `unl_cs_senior_design, unl_cs_senior_design, unl_cs_degree_requirements, unl_cs_major_overview, unl_cs_degree_requirements`

### How old can a student's chosen catalog be at the time of graduation?
- Expected doc: `unl_cs_degree_requirements`
- Dense: Expected document was retrieved but ranked below another source.
- Hybrid: Expected document was retrieved but ranked below another source.
- Dense docs: `unl_engineering_college_overview, unl_cs_degree_requirements, unl_engineering_college_overview, unl_engineering_college_overview, unl_add_drop_withdraw`
- Hybrid docs: `unl_engineering_college_overview, unl_cs_degree_requirements, unl_engineering_college_overview, unl_undergrad_policies_pdf, unl_undergrad_policies_pdf`

### On what basis can scholarships be awarded at Nebraska?
- Expected doc: `unl_scholarships`
- Dense: Expected document was not retrieved.
- Hybrid: Expected document was retrieved but ranked below another source.
- Dense docs: `unl_new_nebraskan_scholarship, unl_engineering_college_overview, unl_new_nebraskan_scholarship, unl_payment_credits_refunds, unl_cost_of_attendance`
- Hybrid docs: `unl_sap_policy, unl_scholarships, unl_scholarships, unl_sap_policy, unl_new_nebraskan_scholarship`

## Regressions

None.
