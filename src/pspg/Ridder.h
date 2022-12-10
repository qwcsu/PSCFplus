#ifndef NUM_ROOT_RIDDER_H
#define NUM_ROOT_RIDDER_H

#include <cmath>
#include <iostream>
#include "System.h"

template <class System, typename T = double>
class Ridder
{

public:
    /**
     * @brief Construct a new Ridder object 
     * 
     */
    Ridder() = default;

    /**
     * @brief Construct a new Ridder object
     * 
     * @param max_itr 
     * @param err 
     */
    Ridder(int max_itr, T err);

    /**
     * @brief Destroy the Ridder object
     * 
     */
    ~Ridder() = default;

    T solve(System sys, T x1, T x2);

    T sign(T a, T b);

private:

    int max_itr_;

    T err_;

    const T UNUSED_ = -1.11e30;

};

template <class System, typename T>
Ridder<System, T>::Ridder(int max_itr, T err)
{
    this->max_itr_ = max_itr;
    this->err_     = err;
    std::cout << "Constructor is called!" << std::endl;
    std::cout << "The maximum steps of iteration is " << max_itr_ << std::endl;
    std::cout << "The error is " << err_ << std::endl;
}

template <class System, typename T>
T
Ridder<System, T>::solve(System sys, T x1, T x2)
{
    T result,
      fh, fl, fm, fnew, 
      xh, xl, xm, xnew,
      s;   
    int j;

    fl = sys.func(x1);
    fh = sys.func(x2);
    
    if ((fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0))
    {
        xl = x1;
        xh = x2;
        result = this->UNUSED_;
        
        for (j = 0; j < this->max_itr_; j++)
        {
            xm = 0.5 * (xl + xh);
            fm = sys.func(xm);
            s  = std::sqrt(fm * fm - fl * fh);

            if (s == 0.0)    
                return result;
            
            xnew = xm + (xm - xl) * ((fl >= fh ? 1.0 : -1.0 )* fm / s);

            if (std::fabs(xnew - result) <= this->err_)
                return result;

            result = xnew;
            fnew = sys.func(result);

            if (fnew == 0.0)
                return result;

            if (sign(fm, fnew) != fm)
            {
                xl = xm;
                fl = fm;
                xh = result;
                fh = fnew;
            }
            else if (sign(fl, fnew) != fl)
            {
                xh = result;
                fh = fnew;
            }
            else if (sign(fh, fnew) != fh)
            {
                xl = result;
                fl = fnew;
            }
            else
            {
                std::cout <<  "Never get there." << std::endl;
                exit(1);
            }

            if (std::fabs(xh - xl) <= this->err_)
                return result;
        }
        std::cout <<  "Exceed maximum iterationas." << std::endl;
        exit(1);
    }
    else
    {
        if (fl == 0.0)
            return x1;
        if (fh == 0.0)
            return x2;
        
        std::cout <<  "The root must be bracketed." << std::endl;
        exit(1);
    }
    return 0.0;
}

template <class System, typename T>
T
Ridder<System, T>::sign(T a, T b)
{
    if (b >= 0.0)
        return std::fabs(a);
    else
        return -std::fabs(a);
}

#endif