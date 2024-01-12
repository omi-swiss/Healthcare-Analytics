SELECT * FROM HEALTHSCHEMA.HEALTH_DATA LIMIT 100;

-- // Univariate Analysis //

--Unique number of hospital codes and their distribution
SELECT HOSPITAL_CODE, COUNT(DISTINCT CASE_ID) AS NUMBER_OF_CASES
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1
ORDER BY 2 DESC;

--Unique number of departments and their distribution
SELECT DEPARTMENT, COUNT(*) AS DEPARTMENT_COUNT
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY DEPARTMENT;

--Average Admission Deposit by Severity of Illness: 
SELECT SEVERITY_OF_ILLNESS, AVG(ADMISSION_DEPOSIT) AS AVG_DEPOSIT
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY SEVERITY_OF_ILLNESS;

--Count of Cases by Type of Admission:
SELECT TYPE_OF_ADMISSION, COUNT(*) AS ADMISSION_COUNT
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY TYPE_OF_ADMISSION;

--Average Number of Visitors by Age Group:
SELECT AGE, AVG(VISITORS_WITH_PATIENT) AS AVG_VISITORS
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY AGE;

--Count of Cases by Bed Grade:
SELECT BED_GRADE, COUNT(*) AS CASES_COUNT
FROM HEALTHSCHEMA.HEALTH_DATA
WHERE BED_GRADE IS NOT NULL
GROUP BY BED_GRADE;

--Distribution of Cases by Hospital Region:
SELECT HOSPITAL_REGION_CODE, COUNT(*) AS REGION_CASES
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY HOSPITAL_REGION_CODE;

--Frequency of Different Ward Types:
SELECT WARD_TYPE, COUNT(*) AS WARD_COUNT
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY WARD_TYPE;

--Hospital type code distribution
SELECT HOSPITAL_TYPE_CODE, COUNT(DISTINCT HOSPITAL_CODE) AS HOSPITAL_CODE_COUNT, COUNT(DISTINCT CASE_ID) AS CASE_COUNT
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1
ORDER BY 2 DESC;

--Unique number of city code hospitals, and their distribution
SELECT CITY_CODE_HOSPITAL, COUNT(DISTINCT HOSPITAL_CODE) AS HOSPITAL_CODE_COUNT,
    COUNT(DISTINCT HOSPITAL_TYPE_CODE) AS HOSPITAL_TYPE_CODE_COUNT,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT
FROM HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1
ORDER BY 2 DESC;

--// Seeing the patient's length of stay changes wrt Hospital type code, city code, and region code //

SELECT HOSPITAL_TYPE_CODE,
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1;

SELECT CITY_CODE_HOSPITAL,
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1;

SELECT HOSPITAL_REGION_CODE,
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1;

-- //Department of admission and its impact on length of stay//

SELECT DEPARTMENT, 
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1;

-- //Department X Hospital_Region_Code and its impact on LOS//

SELECT DEPARTMENT, HOSPITAL_REGION_CODE,
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1,2;

--// Does more available_extra_rooms_in_hospital impact a patients LOS // (insight)

SELECT AVAILABLE_EXTRA_ROOMS_IN_HOSPITAL, 
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1;

 --// Does more type of admission impact a patients LOS // (insight)

SELECT TYPE_OF_ADMISSION, 
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1;

--// Does more severity of illness impact a patient's LOS // (insight)
SELECT SEVERITY_OF_ILLNESS, 
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1;

--// Does age its impact a patient's LOS // (insight)
SELECT AGE, 
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1
ORDER BY 1 ASC;

--// Admission deposit and its relation to LOS//

SELECT DISTINCT ADMISSION_DEPOSIT FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA;

WITH BASE AS (
    SELECT ADMISSION_DEPOSIT,
        CASE WHEN ADMISSION_DEPOSIT <= 3000 THEN '1. less than 3k'
             WHEN ADMISSION_DEPOSIT <= 7000 then '3. Greater than 7k'
        ELSE '2. Between 3K to 7K' end as DEPOSIT_BUCKET,
        ADMISSION_DATE,
        DISCHARGE_DATE,
        CASE_ID
    FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA            
)
--SELECT*FROM BASE;
    
SELECT DEPOSIT_BUCKET,
    MIN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE,DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT
FROM BASE
GROUP BY 1
ORDER BY 1 ASC;


--// Does more visitors come w/patients who have a more sever illness?//

SELECT SEVERITY_OF_ILLNESS,
    COUNT(DISTINCT Case_ID) as CASE_COUNT,
    MIN(VISITORS_WITH_PATIENT) AS MIN_VISTORS,
    MAX(VISITORS_WITH_PATIENT) AS MAX_VISITORS,
    AVG(VISITORS_WITH_PATIENT) AS AVG_VISITORS,
    MEDIAN(VISITORS_WITH_PATIENT) AS MEDIAN_VISITORS
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1
ORDER BY 5 ASC;

--//Is there any differences in the LOS for different WARD_TYPE & WARD_FACILITY_CODE in each DEPARTMENT//

SELECT DEPARTMENT, WARD_TYPE, WARD_FACILITY_CODE,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT,
    MIN(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1,2,3
ORDER BY 1,2,3;


--//Is there any differences in the LOS for different DEPARTMENT, SEVERITY OF ILLNESS in each DEPARTMENT//

SELECT DEPARTMENT, SEVERITY_OF_ILLNESS,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT,
    MIN(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1,2
ORDER BY 1,2;

--// Does Bed grade affects LOS of patients?//

SELECT BED_GRADE,
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT,
    MIN(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1
ORDER BY 1 ASC;

SELECT SEVERITY_OF_ILLNESS, BED_GRADE, 
    COUNT(DISTINCT CASE_ID) AS CASE_COUNT,
    MIN(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MIN_LENGTH_OF_STAY,
    MAX(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MAX_LENGTH_OF_STAY,
    AVG(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS AVG_LENGTH_OF_STAY,
    MEDIAN(DATEDIFF(day, ADMISSION_DATE, DISCHARGE_DATE)) AS MEDIAN_LENGTH_OF_STAY
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1,2
ORDER BY 1,2 ASC;

--//Do more visitors come when younger patients got admitted than older patients. 
SELECT AGE, 
    COUNT(DISTINCT Case_ID) as CASE_COUNT,
    MIN(VISITORS_WITH_PATIENT) AS MIN_VISTORS,
    MAX(VISITORS_WITH_PATIENT) AS MAX_VISITORS,
    AVG(VISITORS_WITH_PATIENT) AS AVG_VISITORS,
    MEDIAN(VISITORS_WITH_PATIENT) AS MEDIAN_VISITORS
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY 1
ORDER BY 1 ASC;


--//What type of illness & admission does majority of patients who are less than 30 eyars of age have and which dept most of them are getting admitted to?//

With BASE AS (
    SELECT * 
    FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
    WHERE AGE IN ('0-10', '20-Nov', '21-30')

),

ILLNESS_N_ADMISSION AS (
    SELECT TYPE_OF_ADMISSION, SEVERITY_OF_ILLNESS,
        COUNT(DISTINCT CASE_ID) AS CASE_COUNT
    FROM BASE
    GROUP BY 1,2
    ORDER BY 1,2
),
DEPARTMENT AS (
    SELECT DEPARTMENT, COUNT(DISTINCT CASE_ID) AS CASE_COUNT
    FROM BASE
    GROUP BY 1
)
--SELECT * FROM BASE; --43,417 cases

--SELECT * FROM ILLNESS_N_ADMISSION;

SELECT * FROM DEPARTMENT;


--// Are patients below 40 years pay more admission_deposit when they got admitted to the hospital?//

WITH BASE AS (

    SELECT *,
        CASE WHEN AGE IN ('0-10', '20-Nov', '21-30', '31-40') THEN 1 ELSE 0 END AS BELOW_40_IND
    FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
)

SELECT BELOW_40_IND, 
    MIN(ADMISSION_DEPOSIT) AS MIN_DEPOSIT,
    MAX(ADMISSION_DEPOSIT) AS MAX_DEPOSIT,
    AVG(ADMISSION_DEPOSIT) AS AVG_DEPOSIT
FROM BASE
GROUP BY 1;

--//Avg Admission Deposit By Hospital And Severity of Illness
SELECT HOSPITAL_TYPE_CODE, SEVERITY_OF_ILLNESS, AVG(ADMISSION_DEPOSIT) AS AVG_DEPOSIT
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY HOSPITAL_TYPE_CODE, SEVERITY_OF_ILLNESS;

--//Count of Cases by hospital region and ward type
SELECT HOSPITAL_REGION_CODE, WARD_TYPE, COUNT(*) AS CASES_COUNT
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY HOSPITAL_REGION_CODE, WARD_TYPE
Order by 1;

--//Avg number of visitors by age group and type of admission
SELECT AGE, TYPE_OF_ADMISSION, AVG(VISITORS_WITH_PATIENT) AS AVG_VISITORS
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
GROUP BY AGE, TYPE_OF_ADMISSION
ORDER BY 1;

--//Distribution of Bed Grades By Department

SELECT DEPARTMENT, BED_GRADE, COUNT(*) AS COUNT
FROM HEALTHDB.HEALTHSCHEMA.HEALTH_DATA
WHERE BED_GRADE IS NOT NULL
GROUP BY DEPARTMENT, BED_GRADE
ORDER BY 1;