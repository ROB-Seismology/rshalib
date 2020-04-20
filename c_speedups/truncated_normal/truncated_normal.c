# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <string.h>

# include "truncated_normal.h"

/******************************************************************************/

double normal_01_cdf ( double x )

/******************************************************************************/
/*
  Purpose:

    NORMAL_01_CDF evaluates the Normal 01 CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    10 February 1999

  Author:

    John Burkardt

  Reference:

    A G Adams,
    Areas Under the Normal Curve,
    Algorithm 39,
    Computer j.,
    Volume 12, pages 197-198, 1969.

  Parameters:

    Input, double X, the argument of the CDF.

    Output, double CDF, the value of the CDF.
*/
{
  double a1 = 0.398942280444;
  double a2 = 0.399903438504;
  double a3 = 5.75885480458;
  double a4 = 29.8213557808;
  double a5 = 2.62433121679;
  double a6 = 48.6959930692;
  double a7 = 5.92885724438;
  double b0 = 0.398942280385;
  double b1 = 3.8052E-08;
  double b2 = 1.00000615302;
  double b3 = 3.98064794E-04;
  double b4 = 1.98615381364;
  double b5 = 0.151679116635;
  double b6 = 5.29330324926;
  double b7 = 4.8385912808;
  double b8 = 15.1508972451;
  double b9 = 0.742380924027;
  double b10 = 30.789933034;
  double b11 = 3.99019417011;
  double cdf;
  double q;
  double y;
/*
  |X| <= 1.28.
*/
  if ( fabs ( x ) <= 1.28 )
  {
    y = 0.5 * x * x;

    q = 0.5 - fabs ( x ) * ( a1 - a2 * y / ( y + a3 - a4 / ( y + a5
      + a6 / ( y + a7 ) ) ) );
/*
  1.28 < |X| <= 12.7
*/
  }
  else if ( fabs ( x ) <= 12.7 )
  {
    y = 0.5 * x * x;

    q = exp ( - y ) * b0 / ( fabs ( x ) - b1
      + b2 / ( fabs ( x ) + b3
      + b4 / ( fabs ( x ) - b5
      + b6 / ( fabs ( x ) + b7
      - b8 / ( fabs ( x ) + b9
      + b10 / ( fabs ( x ) + b11 ) ) ) ) ) );
/*
  12.7 < |X|
*/
  }
  else
  {
    q = 0.0;
  }
/*
  Take account of negative X.
*/
  if ( x < 0.0 )
  {
    cdf = q;
  }
  else
  {
    cdf = 1.0 - q;
  }

  return cdf;
}
/******************************************************************************/

double normal_01_cdf_inv ( double p )

/******************************************************************************/
/*
  Purpose:

    NORMAL_01_CDF_INV inverts the standard normal CDF.

  Discussion:

    The result is accurate to about 1 part in 10^16.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 December 2004

  Author:

    Original FORTRAN77 version by Michael Wichura.
    C version by John Burkardt.

  Reference:

    Michael Wichura,
    The Percentage Points of the Normal Distribution,
    Algorithm AS 241,
    Applied Statistics,
    Volume 37, Number 3, pages 477-484, 1988.

  Parameters:

    Input, double P, the value of the cumulative probability
    densitity function.  0 < P < 1.  If P is outside this range, an
    "infinite" value is returned.

    Output, double NORMAL_01_CDF_INV, the normal deviate value
    with the property that the probability of a standard normal deviate being
    less than or equal to this value is P.
*/
{
  double a[8] = {
    3.3871328727963666080,     1.3314166789178437745e+2,
    1.9715909503065514427e+3,  1.3731693765509461125e+4,
    4.5921953931549871457e+4,  6.7265770927008700853e+4,
    3.3430575583588128105e+4,  2.5090809287301226727e+3 };
  double b[8] = {
    1.0,                       4.2313330701600911252e+1,
    6.8718700749205790830e+2,  5.3941960214247511077e+3,
    2.1213794301586595867e+4,  3.9307895800092710610e+4,
    2.8729085735721942674e+4,  5.2264952788528545610e+3 };
  double c[8] = {
    1.42343711074968357734,     4.63033784615654529590,
    5.76949722146069140550,     3.64784832476320460504,
    1.27045825245236838258,     2.41780725177450611770e-1,
    2.27238449892691845833e-2,  7.74545014278341407640e-4 };
  double const1 = 0.180625;
  double const2 = 1.6;
  double d[8] = {
    1.0,                        2.05319162663775882187,
    1.67638483018380384940,     6.89767334985100004550e-1,
    1.48103976427480074590e-1,  1.51986665636164571966e-2,
    5.47593808499534494600e-4,  1.05075007164441684324e-9 };
  double e[8] = {
    6.65790464350110377720,     5.46378491116411436990,
    1.78482653991729133580,     2.96560571828504891230e-1,
    2.65321895265761230930e-2,  1.24266094738807843860e-3,
    2.71155556874348757815e-5,  2.01033439929228813265e-7 };
  double f[8] = {
    1.0,                        5.99832206555887937690e-1,
    1.36929880922735805310e-1,  1.48753612908506148525e-2,
    7.86869131145613259100e-4,  1.84631831751005468180e-5,
    1.42151175831644588870e-7,  2.04426310338993978564e-15 };
  double q;
  double r;
  double split1 = 0.425;
  double split2 = 5.0;
  double value;

  if ( p <= 0.0 )
  {
    value = -r8_huge ( );
    return value;
  }

  if ( 1.0 <= p )
  {
    value = r8_huge ( );
    return value;
  }

  q = p - 0.5;

  if ( fabs ( q ) <= split1 )
  {
    r = const1 - q * q;
    value = q * r8poly_value ( 8, a, r ) / r8poly_value ( 8, b, r );
  }
  else
  {
    if ( q < 0.0 )
    {
      r = p;
    }
    else
    {
      r = 1.0 - p;
    }

    if ( r <= 0.0 )
    {
      value = r8_huge ( );
    }
    else
    {
      r = sqrt ( - log ( r ) );

      if ( r <= split2 )
      {
        r = r - const2;
        value = r8poly_value ( 8, c, r ) / r8poly_value ( 8, d, r );
       }
       else
       {
         r = r - split2;
         value = r8poly_value ( 8, e, r ) / r8poly_value ( 8, f, r );
      }
    }

    if ( q < 0.0 )
    {
      value = - value;
    }

  }

  return value;
}
/******************************************************************************/

double normal_01_mean ( )

/******************************************************************************/
/*
  Purpose:

    NORMAL_01_MEAN returns the mean of the Normal 01 PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    18 September 2004

  Author:

    John Burkardt

  Parameters:

    Output, double MEAN, the mean of the PDF.
*/
{
  double mean;

  mean = 0.0;

  return mean;
}
/******************************************************************************/

double normal_01_moment ( int order )

/******************************************************************************/
/*
  Purpose:

    NORMAL_01_MOMENT evaluates moments of the Normal 01 PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    31 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, int ORDER, the order of the moment.
    0 <= ORDER.

    Output, double NORMAL_01_MOMENT, the value of the moment.
*/
{
  double value;

  if ( ( order % 2 ) == 0 )
  {
    value = r8_factorial2 ( order - 1 );
  }
  else
  {
    value = 0.0;
  }

  return value;
}
/******************************************************************************/

double normal_01_pdf ( double x )

/******************************************************************************/
/*
  Purpose:

    NORMAL_01_PDF evaluates the Normal 01 PDF.

  Discussion:

    The Normal 01 PDF is also called the "Standard Normal" PDF, or
    the Normal PDF with 0 mean and variance 1.

    PDF(X) = exp ( - 0.5 * X^2 ) / sqrt ( 2 * PI )

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    18 September 2004

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the PDF.

    Output, double PDF, the value of the PDF.
*/
{
  double pdf;
  const double pi = 3.14159265358979323;

  pdf = exp ( -0.5 * x * x ) / sqrt ( 2.0 * pi );

  return pdf;
}
/******************************************************************************/

double normal_01_sample ( int *seed )

/******************************************************************************/
/*
  Purpose:

    NORMAL_01_SAMPLE samples the standard normal probability distribution.

  Discussion:

    The standard normal probability distribution function (PDF) has
    mean 0 and standard deviation 1.

    The Box-Muller method is used, which is efficient, but
    generates two values at a time.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    18 September 2004

  Author:

    John Burkardt

  Parameters:

    Input/output, int *SEED, a seed for the random number generator.

    Output, double NORMAL_01_SAMPLE, a normally distributed random value.
*/
{
  const double pi = 3.14159265358979323;
  double r1;
  double r2;
  static int used = -1;
  double x;
  static double y = 0.0;

  if ( used == -1 )
  {
    used = 0;
  }
/*
  If we've used an even number of values so far, generate two more,
  return one, and save one.
*/
  if ( ( used % 2 )== 0 )
  {
    for ( ; ; )
    {
      r1 = r8_uniform_01 ( seed );
      if ( r1 != 0.0 )
      {
        break;
      }
    }

    r2 = r8_uniform_01 ( seed );

    x = sqrt ( -2.0 * log ( r1 ) ) * cos ( 2.0 * pi * r2 );
    y = sqrt ( -2.0 * log ( r1 ) ) * sin ( 2.0 * pi * r2 );
  }
  else
  {

    x = y;

  }

  used = used + 1;

  return x;
}
/******************************************************************************/

double normal_01_variance ( )

/******************************************************************************/
/*
  Purpose:

    NORMAL_01_VARIANCE returns the variance of the Normal 01 PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    18 September 2004

  Author:

    John Burkardt

  Parameters:

    Output, double VARIANCE, the variance of the PDF.
*/
{
  double variance;

  variance = 1.0;

  return variance;
}
/******************************************************************************/

double normal_cdf ( double x, double a, double b )

/******************************************************************************/
/*
  Purpose:

    NORMAL_CDF evaluates the Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2004

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the CDF.

    Input, double A, B, the parameters of the PDF.
    0.0 < B.

    Output, double CDF, the value of the CDF.
*/
{
  double cdf;
  double y;

  y = ( x - a ) / b;

  cdf = normal_01_cdf ( y );

  return cdf;
}
/******************************************************************************/

double *normal_cdf_array ( double *x, int n, double a, double b )

/******************************************************************************/
/*
  Purpose:

    NORMAL_CDF evaluates the Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2004

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the CDF.

    Input, double A, B, the parameters of the PDF.
    0.0 < B.

    Output, double CDF, the value of the CDF.
*/
{
  double cdf[n];
  int i;

  for (i=0; i < n; i++)
  {
  	cdf[i] = normal_cdf(x[i], a, b);
  }

  return cdf;
}
/******************************************************************************/

double normal_cdf_inv ( double cdf, double a, double b )

/******************************************************************************/
/*
  Purpose:

    NORMAL_CDF_INV inverts the Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2004

  Author:

    John Burkardt

  Reference:

    Algorithm AS 111,
    Applied Statistics,
    Volume 26, pages 118-121, 1977.

  Parameters:

    Input, double CDF, the value of the CDF.
    0.0 <= CDF <= 1.0.

    Input, double A, B, the parameters of the PDF.
    0.0 < B.

    Output, double NORMAL_CDF_INV, the corresponding argument.
*/
{
  double x;
  double x2;

  if ( cdf < 0.0 || 1.0 < cdf )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "NORMAL_CDF_INV - Fatal error!\n" );
    fprintf ( stderr, "  CDF < 0 or 1 < CDF.\n" );
    exit ( 1 );
  }

  x2 = normal_01_cdf_inv ( cdf );

  x = a + b * x2;

  return x;
}
/******************************************************************************/

double normal_mean ( double a, double b )

/******************************************************************************/
/*
  Purpose:

    NORMAL_MEAN returns the mean of the Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2004

  Author:

    John Burkardt

  Parameters:

    Input, double A, B, the parameters of the PDF.
    0.0 < B.

    Output, double MEAN, the mean of the PDF.
*/
{
  double mean;

  mean = a;

  return mean;
}
/******************************************************************************/

double normal_moment ( int order, double mu, double sigma )

/******************************************************************************/
/*
  Purpose:

    NORMAL_MOMENT evaluates moments of the Normal PDF.

  Discussion:

    The formula was posted by John D Cook.

    Order  Moment
    -----  ------
      0    1
      1    mu
      2    mu^2 +         sigma^2
      3    mu^3 +  3 mu   sigma^2
      4    mu^4 +  6 mu^2 sigma^2 +   3      sigma^4
      5    mu^5 + 10 mu^3 sigma^2 +  15 mu   sigma^4
      6    mu^6 + 15 mu^4 sigma^2 +  45 mu^2 sigma^4 +  15      sigma^6
      7    mu^7 + 21 mu^5 sigma^2 + 105 mu^3 sigma^4 + 105 mu   sigma^6
      8    mu^8 + 28 mu^6 sigma^2 + 210 mu^4 sigma^4 + 420 mu^2 sigma^6
           + 105 sigma^8

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, int ORDER, the order of the moment.
    0 <= ORDER.

    Input, double MU, the mean of the distribution.

    Input, double SIGMA, the standard deviation of the distribution.

    Output, double NORMAL_MOMENT, the value of the central moment.
*/
{
  int j;
  int j_hi;
  double value;

  j_hi = ( order / 2 );

  value = 0.0;
  for ( j = 0; j <= j_hi; j++ )
  {
    value = value
      + r8_choose ( order, 2 * j )
      * r8_factorial2 ( 2 * j - 1 )
      * pow ( mu, order - 2 * j ) * pow ( sigma, 2 * j );
  }

  return value;
}
/******************************************************************************/

double normal_moment_central ( int order, double mu, double sigma )

/******************************************************************************/
/*
  Purpose:

    NORMAL_MOMENT_CENTRAL evaluates central moments of the Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, int ORDER, the order of the moment.
    0 <= ORDER.

    Input, double MU, the mean of the distribution.

    Input, double SIGMA, the standard deviation of the distribution.

    Output, double NORMAL_MOMENT_CENTRAL, the value of the central moment.
*/
{
  double value;

  if ( ( order % 2 ) == 0 )
  {
    value = r8_factorial2 ( order - 1 ) * pow ( sigma, order );
  }
  else
  {
    value = 0.0;
  }

  return value;
}
/******************************************************************************/

double normal_moment_central_values ( int order, double mu, double sigma )

/******************************************************************************/
/*
  Purpose:

    NORMAL_MOMENT_CENTRAL_VALUES: moments 0 through 10 of the Normal PDF.

  Discussion:

    The formula was posted by John D Cook.

    Order  Moment
    -----  ------
      0    1
      1    0
      2    sigma^2
      3    0
      4    3 sigma^4
      5    0
      6    15 sigma^6
      7    0
      8    105 sigma^8
      9    0
     10    945 sigma^10

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    31 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, int ORDER, the order of the moment.
    0 <= ORDER <= 10.

    Input, double MU, the mean of the distribution.

    Input, double SIGMA, the standard deviation of the distribution.

    Output, double NORMAL_MOMENT_CENTRAL_VALUES, the value of the central moment.
*/
{
  double value;

  if ( order == 0 )
  {
    value = 1.0;
  }
  else if ( order == 1 )
  {
    value = 0.0;
  }
  else if ( order == 2 )
  {
    value = pow ( sigma, 2 );
  }
  else if ( order == 3 )
  {
    value = 0.0;
  }
  else if ( order == 4 )
  {
    value = 3.0 * pow ( sigma, 4 );
  }
  else if ( order == 5 )
  {
    value = 0.0;
  }
  else if ( order == 6 )
  {
    value = 15.0 * pow ( sigma, 6 );
  }
  else if ( order == 7 )
  {
    value = 0.0;
  }
  else if ( order == 8 )
  {
    value = 105.0 * pow ( sigma, 8 );
  }
  else if ( order == 9 )
  {
    value = 0.0;
  }
  else if ( order == 10 )
  {
    value = 945.0 * pow ( sigma, 10 );
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "NORMAL_MOMENT_CENTRAL_VALUES - Fatal error!\n" );
    fprintf ( stderr, "  Only ORDERS 0 through 10 are available.\n" );
    exit ( 1 );
  }

  return value;
}
/******************************************************************************/

double normal_moment_values ( int order, double mu, double sigma )

/******************************************************************************/
/*
  Purpose:

    NORMAL_MOMENT_VALUES evaluates moments 0 through 8 of the Normal PDF.

  Discussion:

    The formula was posted by John D Cook.

    Order  Moment
    -----  ------
      0    1
      1    mu
      2    mu^2 +         sigma^2
      3    mu^3 +  3 mu   sigma^2
      4    mu^4 +  6 mu^2 sigma^2 +   3      sigma^4
      5    mu^5 + 10 mu^3 sigma^2 +  15 mu   sigma^4
      6    mu^6 + 15 mu^4 sigma^2 +  45 mu^2 sigma^4 +  15      sigma^6
      7    mu^7 + 21 mu^5 sigma^2 + 105 mu^3 sigma^4 + 105 mu   sigma^6
      8    mu^8 + 28 mu^6 sigma^2 + 210 mu^4 sigma^4 + 420 mu^2 sigma^6
           + 105 sigma^8

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    31 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, int ORDER, the order of the moment.
    0 <= ORDER <= 8.

    Input, double MU, the mean of the distribution.

    Input, double SIGMA, the standard deviation of the distribution.

    Output, double NORMAL_MOMENT_VALUES, the value of the central moment.
*/
{
  double value;

  if ( order == 0 )
  {
    value = 1.0;
  }
  else if ( order == 1 )
  {
    value = mu;
  }
  else if ( order == 2 )
  {
    value = pow ( mu, 2 ) + pow ( sigma, 2 );
  }
  else if ( order == 3 )
  {
    value = pow ( mu, 3 ) + 3.0 * mu * pow ( sigma, 2 );
  }
  else if ( order == 4 )
  {
    value = pow ( mu, 4 ) + 6.0 * pow ( mu, 2 ) * pow ( sigma, 2 )
      + 3.0 * pow ( sigma, 4 );
  }
  else if ( order == 5 )
  {
    value = pow ( mu, 5 ) + 10.0 * pow ( mu, 3 ) * pow ( sigma, 2 )
      + 15.0 * mu * pow ( sigma, 4 );
  }
  else if ( order == 6 )
  {
    value = pow ( mu, 6 ) + 15.0 * pow ( mu, 4 ) * pow ( sigma, 2 )
      + 45.0 * pow ( mu, 2 ) * pow ( sigma, 4 )
      + 15.0 * pow ( sigma, 6 );
  }
  else if ( order == 7 )
  {
    value = pow ( mu, 7 ) + 21.0 * pow ( mu, 5 ) * pow ( sigma, 2 )
      + 105.0 * pow ( mu, 3 ) * pow ( sigma, 4 )
      + 105.0 * mu * pow ( sigma, 6 );
  }
  else if ( order == 8 )
  {
    value = pow ( mu, 8 ) + 28.0 * pow ( mu, 6 ) * pow ( sigma, 2 )
      + 210.0 * pow ( mu, 4 ) * pow ( sigma, 4 )
      + 420.0 * pow ( mu, 2 ) * pow ( sigma, 6 ) + 105.0 * pow ( sigma, 8 );
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "NORMAL_MOMENT_VALUES - Fatal error!\n" );
    fprintf ( stderr, "  Only ORDERS 0 through 8 are available.\n" );
    exit ( 1 );
  }

  return value;
}
/******************************************************************************/

double normal_pdf ( double x, double a, double b )

/******************************************************************************/
/*
  Purpose:

    NORMAL_PDF evaluates the Normal PDF.

  Discussion:

    PDF(A,B;X)
      = exp ( - 0.5 * ( ( X - A ) / B )^2 )
      / ( B * SQRT ( 2 * PI ) )

    The normal PDF is also known as the Gaussian PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2004

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the PDF.

    Input, double A, B, the parameters of the PDF.
    0.0 < B.

    Output, double PDF, the value of the PDF.
*/
{
  double pdf;
  const double pi = 3.14159265358979323;
  double y;

  y = ( x - a ) / b;

  pdf = exp ( -0.5 * y * y )  / ( b * sqrt ( 2.0 * pi ) );

  return pdf;
}
/******************************************************************************/

double *normal_pdf_array ( double *x, int n, double a, double b )

/******************************************************************************/
/*
  Purpose:

    NORMAL_PDF evaluates the Normal PDF.

  Discussion:

    PDF(A,B;X)
      = exp ( - 0.5 * ( ( X - A ) / B )^2 )
      / ( B * SQRT ( 2 * PI ) )

    The normal PDF is also known as the Gaussian PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2004

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the PDF.

    Input, double A, B, the parameters of the PDF.
    0.0 < B.

    Output, double PDF, the value of the PDF.
*/
{
  double pdf[n];
  int i;

  for (i=0; i < n; i++)
  {
  	pdf[i] = normal_pdf(x[i], a, b);
  }

  return pdf;
}
/******************************************************************************/

double normal_sample ( double a, double b, int *seed )

/******************************************************************************/
/*
  Purpose:

    NORMAL_SAMPLE samples the Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    20 November 2004

  Author:

    John Burkardt

  Parameters:

    Input, double A, B, the parameters of the PDF.
    0.0 < B.

    Input/output, int *SEED, a seed for the random number generator.

    Output, double NORMAL_SAMPLE, a sample of the PDF.
*/
{
  double x;

  x = normal_01_sample ( seed );

  x = a + b * x;

  return x;
}
/******************************************************************************/

double normal_variance ( double a, double b )

/******************************************************************************/
/*
  Purpose:

    NORMAL_VARIANCE returns the variance of the Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2004

  Author:

    John Burkardt

  Parameters:

    Input, double A, B, the parameters of the PDF.
    0.0 < B.

    Output, double VARIANCE, the variance of the PDF.
*/
{
  double variance;

  variance = b * b;

  return variance;
}
/******************************************************************************/

double r8_abs ( double x )

/******************************************************************************/
/*
  Purpose:

    R8_ABS returns the absolute value of an R8.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 November 2006

  Author:

    John Burkardt

  Parameters:

    Input, double X, the quantity whose absolute value is desired.

    Output, double R8_ABS, the absolute value of X.
*/
{
  double value;

  if ( 0.0 <= x )
  {
    value = x;
  }
  else
  {
    value = -x;
  }
  return value;
}
/******************************************************************************/

double r8_choose ( int n, int k )

/******************************************************************************/
/*
  Purpose:

    R8_CHOOSE computes the binomial coefficient C(N,K) as an R8.

  Discussion:

    The value is calculated in such a way as to avoid overflow and
    roundoff.  The calculation is done in R8 arithmetic.

    The formula used is:

      C(N,K) = N! / ( K! * (N-K)! )

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 July 2008

  Author:

    John Burkardt

  Reference:

    ML Wolfson, HV Wright,
    Algorithm 160:
    Combinatorial of M Things Taken N at a Time,
    Communications of the ACM,
    Volume 6, Number 4, April 1963, page 161.

  Parameters:

    Input, int N, K, the values of N and K.

    Output, double R8_CHOOSE, the number of combinations of N
    things taken K at a time.
*/
{
  int i;
  int mn;
  int mx;
  double value;

  if ( k < n - k )
  {
    mn = k;
    mx = n - k;
  }
  else
  {
    mn = n - k;
    mx = k;
  }

  if ( mn < 0 )
  {
    value = 0.0;
  }
  else if ( mn == 0 )
  {
    value = 1.0;
  }
  else
  {
    value = ( double ) ( mx + 1 );

    for ( i = 2; i <= mn; i++ )
    {
      value = ( value * ( double ) ( mx + i ) ) / ( double ) i;
    }
  }

  return value;
}
/******************************************************************************/

double r8_factorial2 ( int n )

/******************************************************************************/
/*
  Purpose:

    R8_FACTORIAL2 computes the double factorial function.

  Discussion:

    FACTORIAL2( N ) = Product ( N * (N-2) * (N-4) * ... * 2 )  (N even)
                    = Product ( N * (N-2) * (N-4) * ... * 1 )  (N odd)

     N Value
    -- -----
     0     1
     1     1
     2     2
     3     3
     4     8
     5    15
     6    48
     7   105
     8   384
     9   945
    10  3840

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 July 2008

  Author:

    John Burkardt

  Parameters:

    Input, int N, the argument of the double factorial
    function.  If N is less than 1, R8_FACTORIAL2 is returned as 1.0.

    Output, double R8_FACTORIAL2, the value of Factorial2(N).
*/
{
  int n_copy;
  double value;

  value = 1.0;

  if ( n < 1 )
  {
    return value;
  }

  n_copy = n;

  while ( 1 < n_copy )
  {
    value = value * ( double ) n_copy;
    n_copy = n_copy - 2;
  }

  return value;
}
/******************************************************************************/

double r8_huge ( )

/******************************************************************************/
/*
  Purpose:

    R8_HUGE returns a "huge" R8.

  Discussion:

    HUGE_VAL is the largest representable legal real number, and is usually
    defined in math.h, or sometimes in stdlib.h.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    08 May 2003

  Author:

    John Burkardt

  Parameters:

    Output, double R8_HUGE, a "huge" real value.
*/
{
  return HUGE_VAL;
}
/******************************************************************************/

double r8_log_2 ( double x )

/******************************************************************************/
/*
  Purpose:

    R8_LOG_2 returns the logarithm base 2 of an R8.

  Discussion:

    R8_LOG_2 ( X ) = Log ( |X| ) / Log ( 2.0 )

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    12 May 2006

  Author:

    John Burkardt

  Parameters:

    Input, double X, the number whose base 2 logarithm is desired.
    X should not be 0.

    Output, double R8_LOG_2, the logarithm base 2 of the absolute
    value of X.  It should be true that |X| = 2^R_LOG_2.
*/
{
  double value;

  if ( x == 0.0 )
  {
    value = - r8_huge ( );
  }
  else
  {
    value = log ( fabs ( x ) ) / log ( 2.0 );
  }

  return value;
}
/******************************************************************************/

double r8_mop ( int i )

/******************************************************************************/
/*
  Purpose:

    R8_MOP returns the I-th power of -1 as an R8 value.

  Discussion:

    An R8 is an double value.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 July 2008

  Author:

    John Burkardt

  Parameters:

    Input, int I, the power of -1.

    Output, double R8_MOP, the I-th power of -1.
*/
{
  double value;

  if ( ( i % 2 ) == 0 )
  {
    value = + 1.0;
  }
  else
  {
    value = - 1.0;
  }

  return value;
}
/******************************************************************************/

double r8_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    R8_UNIFORM_01 returns a unit pseudorandom R8.

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2^31 - 1 )
      unif = seed / ( 2^31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    11 August 2004

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Springer Verlag, pages 201-202, 1983.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

  Parameters:

    Input/output, int *SEED, the "seed" value.  Normally, this
    value should not be 0.  On output, SEED has been updated.

    Output, double R8_UNIFORM_01, a new pseudorandom variate, strictly between
    0 and 1.
*/
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }
/*
  Although SEED can be represented exactly as a 32 bit integer,
  it generally cannot be represented exactly as a 32 bit real number!
*/
  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}
/******************************************************************************/

double r8poly_value ( int n, double a[], double x )

/******************************************************************************/
/*
  Purpose:

    R8POLY_VALUE evaluates a double precision polynomial.

  Discussion:

    For sanity's sake, the value of N indicates the NUMBER of
    coefficients, or more precisely, the ORDER of the polynomial,
    rather than the DEGREE of the polynomial.  The two quantities
    differ by 1, but cause a great deal of confusion.

    Given N and A, the form of the polynomial is:

      p(x) = a[0] + a[1] * x + ... + a[n-2] * x^(n-2) + a[n-1] * x^(n-1)

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    13 August 2004

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the polynomial.

    Input, double A[N], the coefficients of the polynomial.
    A[0] is the constant term.

    Input, double X, the point at which the polynomial is to be evaluated.

    Output, double R8POLY_VALUE, the value of the polynomial at X.
*/
{
  int i;
  double value;

  value = 0.0;

  for ( i = n-1; 0 <= i; i-- )
  {
    value = value * x + a[i];
  }

  return value;
}
/******************************************************************************/

double r8vec_max ( int n, double *dvec )

/******************************************************************************/
/*
  Purpose:

    R8VEC_MAX returns the value of the maximum element in an R8VEC.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    15 October 1998

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in the array.

    Input, double *RVEC, a pointer to the first entry of the array.

    Output, double R8VEC_MAX, the value of the maximum element.  This
    is set to 0.0 if N <= 0.
*/
{
  int i;
  double rmax;
  double *r8vec_pointer;

  if ( n <= 0 )
  {
    return 0.0;
  }

  for ( i = 0; i < n; i++ )
  {
    if ( i == 0 )
    {
      rmax = *dvec;
      r8vec_pointer = dvec;
    }
    else
    {
      r8vec_pointer++;
      if ( rmax < *r8vec_pointer )
      {
        rmax = *r8vec_pointer;
      }
    }
  }

  return rmax;

}
/******************************************************************************/

double r8vec_mean ( int n, double x[] )

/******************************************************************************/
/*
  Purpose:

    R8VEC_MEAN returns the mean of an R8VEC.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 May 1999

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in the vector.

    Input, double X[N], the vector whose mean is desired.

    Output, double R8VEC_MEAN, the mean, or average, of the vector entries.
*/
{
  int i;
  double mean;

  mean = 0.0;
  for ( i = 0; i < n; i++ )
  {
    mean = mean + x[i];
  }

  mean = mean / ( double ) n;

  return mean;
}
/******************************************************************************/

double r8vec_min ( int n, double *dvec )

/******************************************************************************/
/*
  Purpose:

    R8VEC_MIN returns the value of the minimum element in an R8VEC.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    15 October 1998

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in the array.

    Input, double *RVEC, a pointer to the first entry of the array.

    Output, double R8VEC_MIN, the value of the minimum element.  This
    is set to 0.0 if N <= 0.
*/
{
  int i;
  double rmin;
  double *r8vec_pointer;

  if ( n <= 0 )
  {
    return 0.0;
  }

  for ( i = 0; i < n; i++ )
  {
    if ( i == 0 )
    {
      rmin = *dvec;
      r8vec_pointer = dvec;
    }
    else
    {
      r8vec_pointer++;
      if ( *r8vec_pointer < rmin )
      {
        rmin = *r8vec_pointer;
      }
    }
  }

  return rmin;

}
/******************************************************************************/

double r8vec_variance ( int n, double x[] )

/******************************************************************************/
/*
  Purpose:

    R8VEC_VARIANCE returns the variance of an R8VEC.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 May 1999

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of entries in the vector.

    Input, double X[N], the vector whose variance is desired.

    Output, double R8VEC_VARIANCE, the variance of the vector entries.
*/
{
  int i;
  double mean;
  double variance;

  mean = r8vec_mean ( n, x );

  variance = 0.0;
  for ( i = 0; i < n; i++ )
  {
    variance = variance + ( x[i] - mean ) * ( x[i] - mean );
  }

  if ( 1 < n )
  {
    variance = variance / ( double ) ( n - 1 );
  }
  else
  {
    variance = 0.0;
  }

  return variance;
}
/******************************************************************************/

void timestamp ( void )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
/******************************************************************************/

double truncated_normal_ab_cdf ( double x, double mu, double s, double a,
  double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_CDF evaluates the truncated Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the CDF.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, B, the lower and upper truncation limits.

    Output, double TRUNCATED_NORMAL_AB_CDF, the value of the CDF.
*/
{
  double alpha;
  double alpha_cdf;
  double beta;
  double beta_cdf;
  double cdf;
  double xi;
  double xi_cdf;

  alpha = ( a - mu ) / s;
  beta = ( b - mu ) / s;
  xi = ( x - mu ) / s;

  if ( xi < alpha )
  {
  	return 0.;
  }

  else if ( xi > beta )
  {
  	return 1.;
  }

  else
  {
    alpha_cdf = normal_01_cdf ( alpha );
    beta_cdf = normal_01_cdf ( beta );
    xi_cdf = normal_01_cdf ( xi );

    cdf = ( xi_cdf - alpha_cdf ) / ( beta_cdf - alpha_cdf );

    return cdf;
  }
}
/******************************************************************************/

double *truncated_normal_ab_cdf_array ( double *x, int n, double mu, double s,
  double a, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_CDF evaluates the truncated Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the CDF.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, B, the lower and upper truncation limits.

    Output, double TRUNCATED_NORMAL_AB_CDF, the value of the CDF.
*/
{
  double cdf[n];
  int i;

  for (i=0; i < n; i++)
  {
  	cdf[i] = truncated_normal_ab_cdf(x[i], mu, s, a, b);
  }

  return cdf;
}
/******************************************************************************/

void truncated_normal_ab_cdf_values ( int *n_data, double *mu, double *sigma,
  double *a, double *b, double *x, double *fx )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_CDF_VALUES: values of the Truncated Normal CDF.

  Discussion:

    The Normal distribution, with mean Mu and standard deviation Sigma,
    is truncated to the interval [A,B].

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    13 September 2013

  Author:

    John Burkardt

  Reference:

    Stephen Wolfram,
    The Mathematica Book,
    Fourth Edition,
    Cambridge University Press, 1999,
    ISBN: 0-521-64314-7,
    LC: QA76.95.W65.

  Parameters:

    Input/output, int *N_DATA.  The user sets N_DATA to 0 before the
    first call.  On each call, the routine increments N_DATA by 1, and
    returns the corresponding data; when there is no more data, the
    output value of N_DATA will be 0 again.

    Output, double *MU, the mean of the distribution.

    Output, double *SIGMA, the standard deviation of the distribution.

    Output, double *A, *B, the lower and upper truncation limits.

    Output, double *X, the argument of the function.

    Output, double *FX, the value of the function.
*/
{
# define N_MAX 11

  static double a_vec[N_MAX] = {
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0 };

  static double b_vec[N_MAX] = {
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0 };

  static double fx_vec[N_MAX] = {
    0.3371694242213513,
    0.3685009225506048,
    0.4006444233448185,
    0.4334107066903040,
    0.4665988676496338,
    0.5000000000000000,
    0.5334011323503662,
    0.5665892933096960,
    0.5993555766551815,
    0.6314990774493952,
    0.6628305757786487 };

  static double mu_vec[N_MAX] = {
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0 };

  static double sigma_vec[N_MAX] = {
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0  };

  static double x_vec[N_MAX] = {
     90.0,
     92.0,
     94.0,
     96.0,
     98.0,
    100.0,
    102.0,
    104.0,
    106.0,
    108.0,
    110.0 };

  if ( *n_data < 0 )
  {
    *n_data = 0;
  }

  *n_data = *n_data + 1;

  if ( N_MAX < *n_data )
  {
    *n_data = 0;
    *a = 0.0;
    *b = 0.0;
    *mu = 0.0;
    *sigma = 0.0;
    *x = 0.0;
    *fx = 0.0;
  }
  else
  {
    *a = a_vec[*n_data-1];
    *b = b_vec[*n_data-1];
    *mu = mu_vec[*n_data-1];
    *sigma = sigma_vec[*n_data-1];
    *x = x_vec[*n_data-1];
    *fx = fx_vec[*n_data-1];
  }

  return;
# undef N_MAX
}
/******************************************************************************/

double truncated_normal_ab_cdf_inv ( double cdf, double mu, double s, double a,
  double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_CDF_INV inverts the truncated Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double CDF, the value of the CDF.
    0.0D+00 <= CDF <= 1.0.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, B, the lower and upper truncation limits.

    Output, double TRUNCATED_NORMAL_AB_CDF_INV, the corresponding argument.
*/
{
  double alpha;
  double alpha_cdf;
  double beta;
  double beta_cdf;
  double x;
  double xi;
  double xi_cdf;

  if ( cdf < 0.0 || 1.0 < cdf )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_AB_CDF_INV - Fatal error!\n" );
    fprintf ( stderr, "  CDF < 0 or 1 < CDF.\n" );
    exit ( 1 );
  }

  alpha = ( a - mu ) / s;
  beta = ( b - mu ) / s;

  alpha_cdf = normal_01_cdf ( alpha );
  beta_cdf = normal_01_cdf ( beta );

  xi_cdf = ( beta_cdf - alpha_cdf ) * cdf + alpha_cdf;
  xi = normal_01_cdf_inv ( xi_cdf );

  x = mu + s * xi;

  return x;
}
/******************************************************************************/

double truncated_normal_ab_mean ( double mu, double s, double a, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_MEAN returns the mean of the truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, B, the lower and upper truncation limits.

    Output, double TRUNCATED_NORMAL_AB_MEAN, the mean of the PDF.
*/
{
  double alpha;
  double alpha_cdf;
  double alpha_pdf;
  double beta;
  double beta_cdf;
  double beta_pdf;
  double mean;

  alpha = ( a - mu ) / s;
  beta = ( b - mu ) / s;

  alpha_cdf = normal_01_cdf ( alpha );
  beta_cdf = normal_01_cdf ( beta  );

  alpha_pdf = normal_01_pdf ( alpha );
  beta_pdf = normal_01_pdf ( beta );

  mean = mu + s * ( alpha_pdf - beta_pdf ) / ( beta_cdf - alpha_cdf );

  return mean;
}
/******************************************************************************/

double truncated_normal_ab_moment ( int order, double mu, double s, double a,
  double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_MOMENT: moments of the truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    11 September 2013

  Author:

    John Burkardt

  Reference:

    Phoebus Dhrymes,
    Moments of Truncated Normal Distributions,
    May 2005.

  Parameters:

    Input, int ORDER, the order of the moment.
    0 <= ORDER.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.
    0.0 < S.

    Input, double A, B, the lower and upper truncation limits.
    A < B.

    Output, double TRUNCATED_NORMAL_AB_MOMENT, the moment of the PDF.
*/
{
  double a_h;
  double a_cdf;
  double a_pdf;
  double b_h;
  double b_cdf;
  double b_pdf;
  double ir;
  double irm1;
  double irm2;
  double moment;
  int r;

  if ( order < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n" );
    fprintf ( stderr, "  ORDER < 0.\n" );
    exit ( 1 );
  }

  if ( s <= 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n" );
    fprintf ( stderr, "  S <= 0.\n" );
    exit ( 1 );
  }

  if ( b <= a )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n" );
    fprintf ( stderr, "  B <= A.\n" );
    exit ( 1 );
  }

  a_h = ( a - mu ) / s;
  a_pdf = normal_01_pdf ( a_h );
  a_cdf = normal_01_cdf ( a_h );

  if ( a_cdf == 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n" );
    fprintf ( stderr, "  PDF/CDF ratio fails, A_CDF too small.\n" );
    fprintf ( stderr, "  A_PDF = %g\n", a_pdf );
    fprintf ( stderr, "  A_CDF = %g\n", a_cdf );
    exit ( 1 );
  }

  b_h = ( b - mu ) / s;
  b_pdf = normal_01_pdf ( b_h );
  b_cdf = normal_01_cdf ( b_h );

  if ( b_cdf == 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_AB_MOMENT - Fatal error!\n" );
    fprintf ( stderr, "  PDF/CDF ratio fails, B_CDF too small.\n" );
    fprintf ( stderr, "  B_PDF = %g\n", b_pdf );
    fprintf ( stderr, "  B_CDF = %g\n", b_cdf );
    exit ( 1 );
  }

  moment = 0.0;
  irm2 = 0.0;
  irm1 = 0.0;

  for ( r = 0; r <= order; r++ )
  {
    if ( r == 0 )
    {
      ir = 1.0;
    }
    else if ( r == 1 )
    {
      ir = - ( b_pdf - a_pdf ) / ( b_cdf - a_cdf );
    }
    else
    {
      ir = ( double ) ( r - 1 ) * irm2
        - ( pow ( b_h, r - 1 ) * b_pdf - pow ( a_h, r - 1 ) * a_pdf )
        / ( b_cdf - a_cdf );
    }

    moment = moment + r8_choose ( order, r ) * pow ( mu, order - r )
      * pow ( s, r ) * ir;

    irm2 = irm1;
    irm1 = ir;
  }

  return moment;
}
/******************************************************************************/

double truncated_normal_ab_pdf ( double x, double mu, double s, double a,
  double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_PDF evaluates the truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the PDF.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, B, the lower and upper truncation limits.

    Output, double TRUNCATED_NORMAL_AB_PDF, the value of the PDF.
*/
{
  double alpha;
  double alpha_cdf;
  double beta;
  double beta_cdf;
  double pdf;
  double xi;
  double xi_pdf;

  alpha = ( a - mu ) / s;
  beta = ( b - mu ) / s;
  xi = ( x - mu ) / s;

  if ( xi < alpha || xi > beta )
  {
  	return 0.;
  }

  else
  {
    alpha_cdf = normal_01_cdf ( alpha );
    beta_cdf = normal_01_cdf ( beta );
    xi_pdf = normal_01_pdf ( xi );

   pdf = xi_pdf / ( beta_cdf - alpha_cdf ) / s;

    return pdf;
  }
}
/******************************************************************************/

double *truncated_normal_ab_pdf_array ( double *x, int n, double mu, double s,
  double a, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_PDF evaluates the truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the PDF.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, B, the lower and upper truncation limits.

    Output, double TRUNCATED_NORMAL_AB_PDF, the value of the PDF.
*/
{
  double pdf[n];
  int i;

  for (i=0; i < n; i++)
  {
  	pdf[i] = truncated_normal_ab_pdf(x[i], mu, s, a, b);
  }

  return pdf;
}
/******************************************************************************/

void truncated_normal_ab_pdf_values ( int *n_data, double *mu, double *sigma,
  double *a, double *b, double *x, double *fx )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_PDF_VALUES: values of the Truncated Normal PDF.

  Discussion:

    The Normal distribution, with mean Mu and standard deviation Sigma,
    is truncated to the interval [A,B].

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    13 September 2013

  Author:

    John Burkardt

  Reference:

    Stephen Wolfram,
    The Mathematica Book,
    Fourth Edition,
    Cambridge University Press, 1999,
    ISBN: 0-521-64314-7,
    LC: QA76.95.W65.

  Parameters:

    Input/output, int *N_DATA.  The user sets N_DATA to 0 before the
    first call.  On each call, the routine increments N_DATA by 1, and
    returns the corresponding data; when there is no more data, the
    output value of N_DATA will be 0 again.

    Output, double *MU, the mean of the distribution.

    Output, double *SIGMA, the standard deviation of the distribution.

    Output, double *A, *B, the lower and upper truncation limits.

    Output, double *X, the argument of the function.

    Output, double *FX, the value of the function.
*/
{
# define N_MAX 11

  static double a_vec[N_MAX] = {
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0 };

  static double b_vec[N_MAX] = {
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0 };

  static double fx_vec[N_MAX] = {
    0.01543301171801836,
    0.01588394472270638,
    0.01624375997031919,
    0.01650575046469259,
    0.01666496869385951,
    0.01671838200940538,
    0.01666496869385951,
    0.01650575046469259,
    0.01624375997031919,
    0.01588394472270638,
    0.01543301171801836 };

  static double mu_vec[N_MAX] = {
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0 };

  static double sigma_vec[N_MAX] = {
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0  };

  static double x_vec[N_MAX] = {
     90.0,
     92.0,
     94.0,
     96.0,
     98.0,
    100.0,
    102.0,
    104.0,
    106.0,
    108.0,
    110.0 };

  if ( *n_data < 0 )
  {
    *n_data = 0;
  }

  *n_data = *n_data + 1;

  if ( N_MAX < *n_data )
  {
    *n_data = 0;
    *a = 0.0;
    *b = 0.0;
    *mu = 0.0;
    *sigma = 0.0;
    *x = 0.0;
    *fx = 0.0;
  }
  else
  {
    *a = a_vec[*n_data-1];
    *b = b_vec[*n_data-1];
    *mu = mu_vec[*n_data-1];
    *sigma = sigma_vec[*n_data-1];
    *x = x_vec[*n_data-1];
    *fx = fx_vec[*n_data-1];
  }

  return;
# undef N_MAX
}
/******************************************************************************/

double truncated_normal_ab_sample ( double mu, double s, double a, double b,
  int *seed )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_SAMPLE samples the truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, B, the lower and upper truncation limits.

    Input/output, int *SEED, a seed for the random number
    generator.

    Output, double TRUNCATED_NORMAL_AB_SAMPLE, a sample of the PDF.
*/
{
  double alpha;
  double alpha_cdf;
  double beta;
  double beta_cdf;
  double u;
  double x;
  double xi;
  double xi_cdf;

  alpha = ( a - mu ) / s;
  beta = ( b - mu ) / s;

  alpha_cdf = normal_01_cdf ( alpha );
  beta_cdf = normal_01_cdf ( beta );

  u = r8_uniform_01 ( seed );
  xi_cdf = alpha_cdf + u * ( beta_cdf - alpha_cdf );
  xi = normal_01_cdf_inv ( xi_cdf );

  x = mu + s * xi;

  return x;
}
/******************************************************************************/

double truncated_normal_ab_variance ( double mu, double s, double a, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_AB_VARIANCE returns the variance of the truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, B, the lower and upper truncation limits.

    Output, double TRUNCATED_NORMAL_AB_VARIANCE, the variance of the PDF.
*/
{
  double alpha;
  double alpha_cdf;
  double alpha_pdf;
  double beta;
  double beta_cdf;
  double beta_pdf;
  double variance;

  alpha = ( a - mu ) / s;
  beta = ( b - mu ) / s;

  alpha_pdf = normal_01_pdf ( alpha );
  beta_pdf = normal_01_pdf ( beta );

  alpha_cdf = normal_01_cdf ( alpha );
  beta_cdf = normal_01_cdf ( beta );

  variance = s * s * ( 1.0
    + ( alpha * alpha_pdf - beta * beta_pdf ) / ( beta_cdf - alpha_cdf )
    - pow ( ( alpha_pdf - beta_pdf ) / ( beta_cdf - alpha_cdf ), 2 ) );

  return variance;
}
/******************************************************************************/

double truncated_normal_a_cdf ( double x, double mu, double s, double a )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_CDF evaluates the lower truncated Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the CDF.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, the lower truncation limit.

    Output, double TRUNCATED_NORMAL_A_CDF, the value of the CDF.
*/
{
  double alpha;
  double alpha_cdf;
  double cdf;
  double xi;
  double xi_cdf;

  alpha = ( a - mu ) / s;
  xi = ( x - mu ) / s;

  alpha_cdf = normal_01_cdf ( alpha );
  xi_cdf = normal_01_cdf ( xi );

  cdf = ( xi_cdf - alpha_cdf ) / ( 1.0 - alpha_cdf );

  return cdf;
}
/******************************************************************************/

void truncated_normal_a_cdf_values ( int *n_data, double *mu, double *sigma,
  double *a, double *x, double *fx )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_CDF_VALUES: values of the Lower Truncated Normal CDF.

  Discussion:

    The Normal distribution, with mean Mu and standard deviation Sigma,
    is truncated to the interval [A,+oo).

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 September 2013

  Author:

    John Burkardt

  Reference:

    Stephen Wolfram,
    The Mathematica Book,
    Fourth Edition,
    Cambridge University Press, 1999,
    ISBN: 0-521-64314-7,
    LC: QA76.95.W65.

  Parameters:

    Input/output, int *N_DATA.  The user sets N_DATA to 0 before the
    first call.  On each call, the routine increments N_DATA by 1, and
    returns the corresponding data; when there is no more data, the
    output value of N_DATA will be 0 again.

    Output, double *MU, the mean of the distribution.

    Output, double *SIGMA, the standard deviation of the distribution.

    Output, double *A, the lower truncation limit.

    Output, double *X, the argument of the function.

    Output, double *FX, the value of the function.
*/
{
# define N_MAX 11

  static double a_vec[N_MAX] = {
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0 };

  static double fx_vec[N_MAX] = {
    0.3293202045481688,
    0.3599223134505957,
    0.3913175216041539,
    0.4233210140873113,
    0.4557365629792204,
    0.4883601253415709,
    0.5209836877039214,
    0.5533992365958304,
    0.5854027290789878,
    0.6167979372325460,
    0.6474000461349729 };

  static double mu_vec[N_MAX] = {
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0 };

  static double sigma_vec[N_MAX] = {
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0  };

  static double x_vec[N_MAX] = {
     90.0,
     92.0,
     94.0,
     96.0,
     98.0,
    100.0,
    102.0,
    104.0,
    106.0,
    108.0,
    110.0 };

  if ( *n_data < 0 )
  {
    *n_data = 0;
  }

  *n_data = *n_data + 1;

  if ( N_MAX < *n_data )
  {
    *n_data = 0;
    *a = 0.0;
    *mu = 0.0;
    *sigma = 0.0;
    *x = 0.0;
    *fx = 0.0;
  }
  else
  {
    *a = a_vec[*n_data-1];
    *mu = mu_vec[*n_data-1];
    *sigma = sigma_vec[*n_data-1];
    *x = x_vec[*n_data-1];
    *fx = fx_vec[*n_data-1];
  }

  return;
# undef N_MAX
}
/******************************************************************************/

double truncated_normal_a_cdf_inv ( double cdf, double mu, double s, double a )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_CDF_INV inverts the lower truncated Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double CDF, the value of the CDF.
    0.0D+00 <= CDF <= 1.0.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, the lower truncation limit.

    Output, double TRUNCATED_NORMAL_A_CDF_INV, the corresponding argument.
*/
{
  double alpha;
  double alpha_cdf;
  double x;
  double xi;
  double xi_cdf;

  if ( cdf < 0.0 || 1.0 < cdf )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_A_CDF_INV - Fatal error!\n" );
    fprintf ( stderr, "  CDF < 0 or 1 < CDF.\n" );
    exit ( 1 );
  }

  alpha = ( a - mu ) / s;

  alpha_cdf = normal_01_cdf ( alpha );

  xi_cdf = ( 1.0 - alpha_cdf ) * cdf + alpha_cdf;
  xi = normal_01_cdf_inv ( xi_cdf );

  x = mu + s * xi;

  return x;
}
/******************************************************************************/

double truncated_normal_a_mean ( double mu, double s, double a )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_MEAN returns the mean of the lower truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, the lower truncation limit.

    Output, double TRUNCATED_NORMAL_A_MEAN, the mean of the PDF.
*/
{
  double alpha;
  double alpha_cdf;
  double alpha_pdf;
  double mean;

  alpha = ( a - mu ) / s;

  alpha_cdf = normal_01_cdf ( alpha );

  alpha_pdf = normal_01_pdf ( alpha );

  mean = mu + s * alpha_pdf / ( 1.0 - alpha_cdf );

  return mean;
}
/******************************************************************************/

double truncated_normal_a_moment ( int order, double mu, double s, double a )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_MOMENT: moments of the lower truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    11 September 2013

  Author:

    John Burkardt

  Reference:

    Phoebus Dhrymes,
    Moments of Truncated Normal Distributions,
    May 2005.

  Parameters:

    Input, int ORDER, the order of the moment.
    0 <= ORDER.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, the lower truncation limit.

    Output, double TRUNCATED_NORMAL_A_MOMENT, the moment of the PDF.
*/
{
  double moment;

  moment = r8_mop ( order )
    * truncated_normal_b_moment ( order, - mu, s, - a );

  return moment;
}
/******************************************************************************/

double truncated_normal_a_pdf ( double x, double mu, double s, double a )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_PDF evaluates the lower truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the PDF.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, the lower truncation limit.

    Output, double TRUNCATED_NORMAL_A_PDF, the value of the PDF.
*/
{
  double alpha;
  double alpha_cdf;
  double pdf;
  double xi;
  double xi_pdf;

  alpha = ( a - mu ) / s;
  xi = ( x - mu ) / s;

  alpha_cdf = normal_01_cdf ( alpha );
  xi_pdf = normal_01_pdf ( xi );

  pdf = xi_pdf / ( 1.0 - alpha_cdf ) / s;

  return pdf;
}
/******************************************************************************/

void truncated_normal_a_pdf_values ( int *n_data, double *mu, double *sigma,
  double *a, double *x, double *fx )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_PDF_VALUES: values of the Lower Truncated Normal PDF.

  Discussion:

    The Normal distribution, with mean Mu and standard deviation Sigma,
    is truncated to the interval [A,+oo).

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 September 2013

  Author:

    John Burkardt

  Reference:

    Stephen Wolfram,
    The Mathematica Book,
    Fourth Edition,
    Cambridge University Press, 1999,
    ISBN: 0-521-64314-7,
    LC: QA76.95.W65.

  Parameters:

    Input/output, int *N_DATA.  The user sets N_DATA to 0 before the
    first call.  On each call, the routine increments N_DATA by 1, and
    returns the corresponding data; when there is no more data, the
    output value of N_DATA will be 0 again.

    Output, double *MU, the mean of the distribution.

    Output, double *SIGMA, the standard deviation of the distribution.

    Output, double *A, the lower truncation limit.

    Output, double *X, the argument of the function.

    Output, double *FX, the value of the function.
*/
{
# define N_MAX 11

  static double a_vec[N_MAX] = {
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0,
     50.0 };

  static double fx_vec[N_MAX] = {
     0.01507373507401876,
     0.01551417047139894,
     0.01586560931024694,
     0.01612150073158793,
     0.01627701240029317,
     0.01632918226724295,
     0.01627701240029317,
     0.01612150073158793,
     0.01586560931024694,
     0.01551417047139894,
     0.01507373507401876 };

  static double mu_vec[N_MAX] = {
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0 };

  static double sigma_vec[N_MAX] = {
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0  };

  static double x_vec[N_MAX] = {
     90.0,
     92.0,
     94.0,
     96.0,
     98.0,
    100.0,
    102.0,
    104.0,
    106.0,
    108.0,
    110.0 };

  if ( *n_data < 0 )
  {
    *n_data = 0;
  }

  *n_data = *n_data + 1;

  if ( N_MAX < *n_data )
  {
    *n_data = 0;
    *a = 0.0;
    *mu = 0.0;
    *sigma = 0.0;
    *x = 0.0;
    *fx = 0.0;
  }
  else
  {
    *a = a_vec[*n_data-1];
    *mu = mu_vec[*n_data-1];
    *sigma = sigma_vec[*n_data-1];
    *x = x_vec[*n_data-1];
    *fx = fx_vec[*n_data-1];
  }

  return;
# undef N_MAX
}
/******************************************************************************/

double truncated_normal_a_sample ( double mu, double s, double a, int *seed )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_SAMPLE samples the lower truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, the lower truncation limit.

    Input/output, int *SEED, a seed for the random number
    generator.

    Output, double TRUNCATED_NORMAL_A_SAMPLE, a sample of the PDF.
*/
{
  double alpha;
  double alpha_cdf;
  double u;
  double x;
  double xi;
  double xi_cdf;

  alpha = ( a - mu ) / s;

  alpha_cdf = normal_01_cdf ( alpha );

  u = r8_uniform_01 ( seed );
  xi_cdf = alpha_cdf + u * ( 1.0 - alpha_cdf );
  xi = normal_01_cdf_inv ( xi_cdf );

  x = mu + s * xi;

  return x;
}
/******************************************************************************/

double truncated_normal_a_variance ( double mu, double s, double a )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_A_VARIANCE: variance of the lower truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double A, the lower truncation limit.

    Output, double TRUNCATED_NORMAL_A_VARIANCE, the variance of the PDF.
*/
{
  double alpha;
  double alpha_cdf;
  double alpha_pdf;
  double variance;

  alpha = ( a - mu ) / s;

  alpha_pdf = normal_01_pdf ( alpha );

  alpha_cdf = normal_01_cdf ( alpha );

  variance = s * s * ( 1.0
    + alpha * alpha_pdf / ( 1.0 - alpha_cdf )
    - pow ( alpha_pdf / ( 1.0 - alpha_cdf ), 2 ) );

  return variance;
}
/******************************************************************************/

double truncated_normal_b_cdf ( double x, double mu, double s, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_CDF evaluates the upper truncated Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the CDF.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double B, the upper truncation limit.

    Output, double TRUNCATED_NORMAL_B_CDF, the value of the CDF.
*/
{
  double beta;
  double beta_cdf;
  double cdf;
  double xi;
  double xi_cdf;

  beta = ( b - mu ) / s;
  xi = ( x - mu ) / s;

  beta_cdf = normal_01_cdf ( beta );
  xi_cdf = normal_01_cdf ( xi );

  cdf = xi_cdf / beta_cdf;

  return cdf;
}
/******************************************************************************/

void truncated_normal_b_cdf_values ( int *n_data, double *mu, double *sigma,
  double *b, double *x, double *fx )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_CDF_VALUES: values of the Upper Truncated Normal CDF.

  Discussion:

    The Normal distribution, with mean Mu and standard deviation Sigma,
    is truncated to the interval [A,B].

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 September 2013

  Author:

    John Burkardt

  Reference:

    Stephen Wolfram,
    The Mathematica Book,
    Fourth Edition,
    Cambridge University Press, 1999,
    ISBN: 0-521-64314-7,
    LC: QA76.95.W65.

  Parameters:

    Input/output, int *N_DATA.  The user sets N_DATA to 0 before the
    first call.  On each call, the routine increments N_DATA by 1, and
    returns the corresponding data; when there is no more data, the
    output value of N_DATA will be 0 again.

    Output, double *MU, the mean of the distribution.

    Output, double *SIGMA, the standard deviation of the distribution.

    Output, double *B, the upper truncation limit.

    Output, double *X, the argument of the function.

    Output, double *FX, the value of the function.
*/
{
# define N_MAX 11

  static double b_vec[N_MAX] = {
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0 };

  static double fx_vec[N_MAX] = {
    0.3525999538650271,
    0.3832020627674540,
    0.4145972709210122,
    0.4466007634041696,
    0.4790163122960786,
    0.5116398746584291,
    0.5442634370207796,
    0.5766789859126887,
    0.6086824783958461,
    0.6400776865494043,
    0.6706797954518312 };

  static double mu_vec[N_MAX] = {
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0 };

  static double sigma_vec[N_MAX] = {
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0  };

  static double x_vec[N_MAX] = {
     90.0,
     92.0,
     94.0,
     96.0,
     98.0,
    100.0,
    102.0,
    104.0,
    106.0,
    108.0,
    110.0 };

  if ( *n_data < 0 )
  {
    *n_data = 0;
  }

  *n_data = *n_data + 1;

  if ( N_MAX < *n_data )
  {
    *n_data = 0;
    *b = 0.0;
    *mu = 0.0;
    *sigma = 0.0;
    *x = 0.0;
    *fx = 0.0;
  }
  else
  {
    *b = b_vec[*n_data-1];
    *mu = mu_vec[*n_data-1];
    *sigma = sigma_vec[*n_data-1];
    *x = x_vec[*n_data-1];
    *fx = fx_vec[*n_data-1];
  }

  return;
# undef N_MAX
}
/******************************************************************************/

double truncated_normal_b_cdf_inv ( double cdf, double mu, double s, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_CDF_INV inverts the upper truncated Normal CDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double CDF, the value of the CDF.
    0.0D+00 <= CDF <= 1.0.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double B, the upper truncation limit.

    Output, double TRUNCATED_NORMAL_B_CDF_INV, the corresponding argument.
*/
{
  double beta;
  double beta_cdf;
  double x;
  double xi;
  double xi_cdf;

  if ( cdf < 0.0 || 1.0 < cdf )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_B_CDF_INV - Fatal error!\n" );
    fprintf ( stderr, "  CDF < 0 or 1 < CDF.\n" );
    exit ( 1 );
  }

  beta = ( b - mu ) / s;

  beta_cdf = normal_01_cdf ( beta );

  xi_cdf = beta_cdf * cdf;
  xi = normal_01_cdf_inv ( xi_cdf );

  x = mu + s * xi;

  return x;
}
/******************************************************************************/

double truncated_normal_b_mean ( double mu, double s, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_MEAN returns the mean of the upper truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double B, the upper truncation limit.

    Output, double TRUNCATED_NORMAL_B_MEAN, the mean of the PDF.
*/
{
  double beta;
  double beta_cdf;
  double beta_pdf;
  double mean;

  beta = ( b - mu ) / s;

  beta_cdf = normal_01_cdf ( beta  );

  beta_pdf = normal_01_pdf ( beta );

  mean = mu - s * beta_pdf / beta_cdf;

  return mean;
}
/******************************************************************************/

double truncated_normal_b_moment ( int order, double mu, double s, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_MOMENT: moments of the upper truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    11 September 2013

  Author:

    John Burkardt

  Reference:

    Phoebus Dhrymes,
    Moments of Truncated Normal Distributions,
    May 2005.

  Parameters:

    Input, int ORDER, the order of the moment.
    0 <= ORDER.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double B, the upper truncation limit.

    Output, double TRUNCATED_NORMAL_B_MOMENT, the moment of the PDF.
*/
{
  double f;
  double h;
  double h_cdf;
  double h_pdf;
  double ir;
  double irm1;
  double irm2;
  double moment;
  int r;

  if ( order < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_B_MOMENT - Fatal error!\n" );
    fprintf ( stderr, "  ORDER < 0.\n" );
    exit ( 1 );
  }

  h = ( b - mu ) / s;
  h_pdf = normal_01_pdf ( h );
  h_cdf = normal_01_cdf ( h );

  if ( h_cdf == 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRUNCATED_NORMAL_B_MOMENT - Fatal error!\n" );
    fprintf ( stderr, "  CDF((B-MU)/S) = 0.\n" );
    exit ( 1 );
  }

  f = h_pdf / h_cdf;

  moment = 0.0;
  irm2 = 0.0;
  irm1 = 0.0;

  for ( r = 0; r <= order; r++ )
  {
    if ( r == 0 )
    {
      ir = 1.0;
    }
    else if ( r == 1 )
    {
      ir = - f;
    }
    else
    {
      ir = - pow ( h, r - 1 ) * f + ( double ) ( r - 1 ) * irm2;
    }

    moment = moment + r8_choose ( order, r ) * pow ( mu, order - r )
      * pow ( s, r ) * ir;

    irm2 = irm1;
    irm1 = ir;
  }

  return moment;
}
/******************************************************************************/

double truncated_normal_b_pdf ( double x, double mu, double s, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_PDF evaluates the upper truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double X, the argument of the PDF.

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double B, the upper truncation limit.

    Output, double TRUNCATED_NORMAL_B_PDF, the value of the PDF.
*/
{
  double beta;
  double beta_cdf;
  double pdf;
  double xi;
  double xi_pdf;

  beta = ( b - mu ) / s;
  xi = ( x - mu ) / s;

  beta_cdf = normal_01_cdf ( beta );
  xi_pdf = normal_01_pdf ( xi );

  pdf = xi_pdf / beta_cdf / s;

  return pdf;
}
/******************************************************************************/

void truncated_normal_b_pdf_values ( int *n_data, double *mu, double *sigma,
  double *b, double *x, double *fx )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_PDF_VALUES: values of the Upper Truncated Normal PDF.

  Discussion:

    The Normal distribution, with mean Mu and standard deviation Sigma,
    is truncated to the interval (-oo,B].

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    14 September 2013

  Author:

    John Burkardt

  Reference:

    Stephen Wolfram,
    The Mathematica Book,
    Fourth Edition,
    Cambridge University Press, 1999,
    ISBN: 0-521-64314-7,
    LC: QA76.95.W65.

  Parameters:

    Input/output, int *N_DATA.  The user sets N_DATA to 0 before the
    first call.  On each call, the routine increments N_DATA by 1, and
    returns the corresponding data; when there is no more data, the
    output value of N_DATA will be 0 again.

    Output, double *MU, the mean of the distribution.

    Output, double *SIGMA, the standard deviation of the distribution.

    Output, double *B, the upper truncation limit.

    Output, double *X, the argument of the function.

    Output, double *FX, the value of the function.
*/
{
# define N_MAX 11

  static double b_vec[N_MAX] = {
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0,
     150.0 };

  static double fx_vec[N_MAX] = {
    0.01507373507401876,
    0.01551417047139894,
    0.01586560931024694,
    0.01612150073158793,
    0.01627701240029317,
    0.01632918226724295,
    0.01627701240029317,
    0.01612150073158793,
    0.01586560931024694,
    0.01551417047139894,
    0.01507373507401876 };

  static double mu_vec[N_MAX] = {
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0,
     100.0 };

  static double sigma_vec[N_MAX] = {
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0  };

  static double x_vec[N_MAX] = {
     90.0,
     92.0,
     94.0,
     96.0,
     98.0,
    100.0,
    102.0,
    104.0,
    106.0,
    108.0,
    110.0 };

  if ( *n_data < 0 )
  {
    *n_data = 0;
  }

  *n_data = *n_data + 1;

  if ( N_MAX < *n_data )
  {
    *n_data = 0;
    *b = 0.0;
    *mu = 0.0;
    *sigma = 0.0;
    *x = 0.0;
    *fx = 0.0;
  }
  else
  {
    *b = b_vec[*n_data-1];
    *mu = mu_vec[*n_data-1];
    *sigma = sigma_vec[*n_data-1];
    *x = x_vec[*n_data-1];
    *fx = fx_vec[*n_data-1];
  }

  return;
# undef N_MAX
}
/******************************************************************************/

double truncated_normal_b_sample ( double mu, double s, double b, int *seed )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_SAMPLE samples the upper truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double B, the upper truncation limit.

    Input/output, int *SEED, a seed for the random number
    generator.

    Output, double TRUNCATED_NORMAL_B_SAMPLE, a sample of the PDF.
*/
{
  double beta;
  double beta_cdf;
  double u;
  double x;
  double xi;
  double xi_cdf;

  beta = ( b - mu ) / s;

  beta_cdf = normal_01_cdf ( beta );

  u = r8_uniform_01 ( seed );
  xi_cdf = u * beta_cdf;
  xi = normal_01_cdf_inv ( xi_cdf );

  x = mu + s * xi;

  return x;
}
/******************************************************************************/

double truncated_normal_b_variance ( double mu, double s, double b )

/******************************************************************************/
/*
  Purpose:

    TRUNCATED_NORMAL_B_VARIANCE: variance of the upper truncated Normal PDF.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 August 2013

  Author:

    John Burkardt

  Parameters:

    Input, double MU, S, the mean and standard deviation of the
    parent Normal distribution.

    Input, double B, the upper truncation limit.

    Output, double TRUNCATED_NORMAL_B_VARIANCE, the variance of the PDF.
*/
{
  double beta;
  double beta_cdf;
  double beta_pdf;
  double variance;

  beta = ( b - mu ) / s;

  beta_pdf = normal_01_pdf ( beta );

  beta_cdf = normal_01_cdf ( beta );

  variance = s * s * ( 1.0
    - beta * beta_pdf / beta_cdf
    - pow ( beta_pdf / beta_cdf, 2 ) );

  return variance;
}
