# HR Analytics Dashboard

**Course:** Visualization & Storytelling Using Tableau
**Institution:** Berlin School of Business and Innovation
**Year:** 2025
**Tools:** Tableau, IBM HR Analytics Dataset

-----

## Project Overview

This project involves the design and development of an interactive HR analytics dashboard using Tableau. The dashboard enables HR executives to monitor key workforce metrics, identify employees at risk of leaving, and evaluate the effectiveness of retention and diversity initiatives.

The project demonstrates how data-driven decision making can transform HR from a reactive reporting function into a proactive strategic partner.

-----

## Dataset

**Source:** IBM HR Analytics Employee Attrition and Performance Dataset (Kaggle)

* 1,470 employee records
* 35 attributes covering demographics, job roles, compensation, satisfaction scores, and attrition status
* Binary Attrition indicator (Yes/No) as the primary outcome variable

-----

## Dashboard Components

**1. Attrition Overview KPIs**

* Total employees: 1,470
* Overall attrition rate: 16%
* High risk employees: 218
* Average monthly income: $6,503

**2. Attrition by Department**

* Sales: 20.6% (highest)
* HR: 19.1%
* R&D: 13.8% (lowest)

**3. Job Satisfaction vs Monthly Income**

* Scatter plot with trend line
* Shows weak positive correlation between salary and satisfaction
* Employees who left cluster at low satisfaction regardless of income

**4. Tenure Analysis**

* Two critical attrition peaks identified: first year and five year mark
* Informs targeted onboarding and career development strategies

**5. Diversity Overview**

* Gender distribution across job roles
* Highlights potential DE&I gaps particularly in managerial positions

**6. Performance Distribution**

* Histogram of performance ratings across the organisation

-----

## Custom Calculated Fields

|Field               |Formula                                          |Purpose                       |
|--------------------|-------------------------------------------------|------------------------------|
|Annual Salary       |Monthly Income x 12                              |Annualised compensation view  |
|Attrition Rate (LOD)|FIXED Department: SUM attrition / COUNT employees|Department level attrition    |
|High Risk Flag      |Job Satisfaction <= 2 AND Years < 3              |Identify flight risk employees|
|Tenure Category     |Early, Mid, Established groupings                |Segment employees by tenure   |

-----

## Key Findings

* Sales department requires urgent targeted retention intervention
* Non-monetary factors drive attrition more than salary levels
* First year and five year tenure marks are the highest risk periods
* Gender imbalance exists in managerial and senior roles

-----

## Skills Demonstrated

Tableau Dashboard Design, Data Visualisation, Storytelling with Data, LOD Expressions, Calculated Fields, Interactive Filters, KPI Design, HR Analytics, Workforce Intelligence
