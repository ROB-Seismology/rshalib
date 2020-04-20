double normal_01_cdf ( double x );
double normal_01_cdf_inv ( double cdf );
double normal_01_mean ( );
double normal_01_moment ( int order );
double normal_01_pdf ( double x );
double normal_01_sample ( int *seed );
double normal_01_variance ( );

double normal_cdf ( double x, double a, double b );
double *normal_cdf_array ( double *x, int n, double a, double b );
double normal_cdf_inv ( double cdf, double a, double b );
double normal_mean ( double a, double b );
double normal_moment ( int order, double mu, double sigma );
double normal_moment_central ( int order, double mu, double sigma );
double normal_moment_central_values ( int order, double mu, double sigma );
double normal_moment_values ( int order, double mu, double sigma );
double normal_pdf ( double x, double a, double b );
double *normal_pdf_array ( double *x, int n, double a, double b );
double normal_sample ( double a, double b, int *seed );
double normal_variance ( double a, double b );

double r8_abs ( double x );
double r8_choose ( int n, int k );
double r8_factorial2 ( int n );
double r8_huge ( );
double r8_log_2 ( double x );
double r8_mop ( int i );
double r8_uniform_01 ( int *seed );

double r8poly_value ( int n, double a[], double x );

double r8vec_max ( int n, double x[] );
double r8vec_mean ( int n, double x[] );
double r8vec_min ( int n, double x[] );
double r8vec_variance ( int n, double x[] );

void timestamp ( );

double truncated_normal_ab_cdf ( double x, double mu, double s, double a,
  double b );
double *truncated_normal_ab_cdf_array ( double *x, int n, double mu, double s,
  double a, double b );
void truncated_normal_ab_cdf_values ( int *n_data, double *mu, double *sigma,
  double *a, double *b, double *x, double *fx );
double truncated_normal_ab_cdf_inv ( double cdf, double mu, double s, double a,
  double b );
double truncated_normal_ab_mean ( double mu, double s, double a, double b );
double truncated_normal_ab_moment ( int order, double mu, double s, double a, double b );
double truncated_normal_ab_pdf ( double x, double mu, double s, double a,
  double b );
double *truncated_normal_ab_pdf_array ( double *x, int n, double mu, double s,
  double a, double b );
void truncated_normal_ab_pdf_values ( int *n_data, double *mu, double *sigma,
  double *a, double *b, double *x, double *fx );
double truncated_normal_ab_sample ( double mu, double s, double a, double b,
  int *seed );
double truncated_normal_ab_variance ( double mu, double s, double a, double b );

double truncated_normal_a_cdf ( double x, double mu, double s, double a );
void truncated_normal_a_cdf_values ( int *n_data, double *mu, double *sigma,
  double *a, double *x, double *fx );
double truncated_normal_a_cdf_inv ( double cdf, double mu, double s, double a );
double truncated_normal_a_mean ( double mu, double s, double a );
double truncated_normal_a_moment ( int order, double mu, double s, double a );
double truncated_normal_a_pdf ( double x, double mu, double s, double a );
void truncated_normal_a_pdf_values ( int *n_data, double *mu, double *sigma,
  double *a, double *x, double *fx );
double truncated_normal_a_sample ( double mu, double s, double a, int *seed );
double truncated_normal_a_variance ( double mu, double s, double a );

double truncated_normal_b_cdf ( double x, double mu, double s, double b );
void truncated_normal_b_cdf_values ( int *n_data, double *mu, double *sigma,
  double *b, double *x, double *fx );
double truncated_normal_b_cdf_inv ( double cdf, double mu, double s, double b );
double truncated_normal_b_mean ( double mu, double s, double b );
double truncated_normal_b_moment ( int order, double mu, double s, double b );
double truncated_normal_b_pdf ( double x, double mu, double s, double b );
void truncated_normal_b_pdf_values ( int *n_data, double *mu, double *sigma,
  double *b, double *x, double *fx );
double truncated_normal_b_sample ( double mu, double s, double b, int *seed );
double truncated_normal_b_variance ( double mu, double s, double b );
