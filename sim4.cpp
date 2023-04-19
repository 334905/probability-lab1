#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <format>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

template <class T>
concept copy_constructible =
        std::constructible_from<T, T&> && std::convertible_to<T&, T> && std::constructible_from<T, const T&>
        && std::convertible_to<const T&, T> && std::constructible_from<T, const T> && std::convertible_to<const T, T>;

template <class T>
concept copy_assignable =
        std::assignable_from<T&, T&> && std::assignable_from<T&, const T&> && std::assignable_from<T&, const T>;

template <class P, class D>
concept distribution_param_type =
        copy_constructible<typename D::param_type> && copy_assignable<typename D::param_type>
        && std::equality_comparable<P>
        && requires {
               typename P::distribution_type;
               requires std::same_as<typename P::distribution_type, D>;
           };

template <class D>
concept random_number_distribution =
        copy_constructible<D> && copy_assignable<D> && std::equality_comparable<D>
        && requires(
                D d,
                const D x,
                typename D::param_type p,
                std::ranlux24 g,
                std::basic_istream<char> is,
                std::basic_ostream<char> os) {
               typename D::result_type;
               requires std::integral<typename D::result_type> || std::floating_point<typename D::result_type>;
               typename D::param_type;
               requires distribution_param_type<typename D::param_type, D>;
               D();
               D(p);
               {
                   d.reset()
               } -> std::same_as<void>;
               {
                   x.param()
               } -> std::same_as<typename D::param_type>;
               {
                   d.param(p)
               } -> std::same_as<void>;
               {
                   d(g)
               } -> std::same_as<typename D::result_type>;
               {
                   d(g, p)
               } -> std::same_as<typename D::result_type>;
               {
                   x.min()
               } -> std::same_as<typename D::result_type>;
               {
                   x.max()
               } -> std::same_as<typename D::result_type>;
               {
                   os << d
               } -> std::same_as<decltype(os)&>;
               {
                   is >> d
               } -> std::same_as<decltype(is)&>;
           };

static_assert(random_number_distribution<std::binomial_distribution<std::size_t>>);

template <std::totally_ordered value_type>
class in_range
{
private:
    value_type from, to;

public:
    in_range(const value_type& from, const value_type& to) : from(from), to(to) {}

    bool operator()(const value_type& value) const
    {
        return from <= value && value < to;
    }
};

template <random_number_distribution distribution, std::predicate<typename distribution::result_type> predicate>
std::uint64_t variable_in_range(
        std::uniform_random_bit_generator auto& gen,
        distribution& distr,
        const predicate& p,
        std::uint64_t series)
{
    std::uint64_t count = 0;
    while (series-- > 0)
    {
        if (p(distr(gen)))
        {
            count++;
        }
    }
    return count;
}

int main()
{
    std::cout.precision(20);

    std::array<std::uint64_t, 4> ns = {
            {10, 100, 1'000, 10'000}
    };
    std::array<double, 5> ps = {
            {0.001, 0.01, 0.1, 0.25, 0.5}
    };

    std::vector<std::thread> threads;
    std::mutex in_lock;

    for (std::size_t i = 0; i < ns.size(); i++)
    {
        std::uint64_t n = ns[i];
        for (std::size_t j = 0; j < ps.size(); j++)
        {
            double p = ps[j];

            threads.emplace_back(
                    [i, j, n, p, &in_lock]() -> void
                    {
                        std::mt19937_64 engine;
                        std::uint64_t series = 1'000'000'000;
                        std::uint64_t lower_bound =
                                static_cast<std::uint64_t>(std::ceil(n / 2.0L - std::sqrt(n * p * (1.0L - p))));
                        std::uint64_t upper_bound =
                                static_cast<std::uint64_t>(std::ceil(n / 2.0L + std::sqrt(n * p * (1.0L - p))));
                        std::binomial_distribution distr(n, p);
                        std::uint64_t positive =
                                variable_in_range(engine, distr, in_range(lower_bound, upper_bound), series);

                        std::unique_lock lck(in_lock);
                        std::format_to(
                                std::ostream_iterator<char>(std::cout),
                                "{:>6} {:>6.3}:    {:>10}/{:<10}\n",
                                n,
                                p,
                                positive,
                                series);
                        std::cout << std::flush;
                    });
        }
    }

    for (std::thread& thread : threads)
    {
        thread.join();
    }
    return 0;
}
