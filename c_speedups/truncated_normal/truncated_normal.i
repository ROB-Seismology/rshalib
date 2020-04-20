/* truncated_normal.i */
%module truncated_normal
%{
/* Put header files here or function declarations like below */
double normal_cdf(double x, double a, double b);
double *normal_cdf_array(double *x, int n, double a, double b);
double normal_pdf(double x, double a, double b);
double *normal_pdf_array(double *x, int n, double a, double b);
double normal_cdf_inv(double cdf, double a, double b);
double truncated_normal_ab_cdf(double x, double mu, double s, double a, double b);
double *truncated_normal_ab_cdf_array(double *x, int n, double mu, double s, double a, double b);
double truncated_normal_ab_pdf(double x, double mu, double s, double a, double b);
double *truncated_normal_ab_pdf_array(double *x, int n, double mu, double s, double a, double b);
double truncated_normal_ab_cdf_inv(double cdf, double mu, double s, double a, double b);
%}

double normal_cdf(double x, double a, double b);
double *normal_cdf_array(double *x, int n, double a, double b);
double normal_pdf(double x, double a, double b);
double *normal_pdf_array(double *x, int n, double a, double b);
double normal_cdf_inv(double cdf, double a, double b);
double truncated_normal_ab_cdf(double x, double mu, double s, double a, double b);
double *truncated_normal_ab_cdf_array(double *x, int n, double mu, double s, double a, double b);
double truncated_normal_ab_pdf(double x, double mu, double s, double a, double b);
double *truncated_normal_ab_pdf_array(double *x, int n, double mu, double s, double a, double b);
double truncated_normal_ab_cdf_inv(double cdf, double mu, double s, double a, double b);
