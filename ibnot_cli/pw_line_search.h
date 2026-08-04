#ifndef _PW_LINE_SEARCH_H_
#define _PW_LINE_SEARCH_H_

#include "line_search.h"

//------------//
// CWCWeights //
//------------//

template <class Scene, class T>
class CLSWeights : public CLineSearch<T, T>
{
protected:
    Scene* m_scene;
    unsigned m_nb;
    
public:
    CLSWeights(Scene* scene,
              const unsigned max_iters,
              const double max_alpha)
    : CLineSearch<T, T>(max_iters, max_alpha)
    {
        m_scene = scene;
        m_nb = m_scene->count_visible_sites();
    }
    
    double compute_function() const
    {
        return m_scene->compute_wcvt_energy();
    }
    
    void compute_gradient(std::vector<T>& V) const
    {
        m_scene->compute_weight_gradient(V);
    }
    
    bool update_scene(const std::vector<T>& X)
    {
        if (m_scene->connectivity_fixed())
        {
            m_scene->clean_pixels();
            m_scene->assign_pixels();
            return true;
        }
        
        // `false` is CORRECT and deliberate: it reassigns EVERY vertex in m_vertices, including any
        // that a trial step has just hidden, which is what allows a hidden vertex to come back. It
        // relies on the invariant that nothing is hidden when the search STARTS -- only then is the
        // visible-only X from collect_visible_weights() also the all-vertices array.
        //
        // Passing true here instead is WRONG: a vertex hidden by one trial would keep its stale
        // weight forever and stay hidden. Measured on apple_0_airplane: 1024 sites -> 395.
        //
        // When the invariant is violated (a previous failed search left vertices hidden) X is SHORT,
        // and update_weights would index past its end. Scene::check_update_size now refuses that
        // update and reports it rather than reading heap garbage.
        m_scene->update_weights(X, false);
        return m_scene->update_triangulation(true);
        //return has_same_vertices();
    }
    
    bool has_same_vertices() const
    {
        unsigned nb = m_scene->count_visible_sites();
        if (nb != m_nb) std::cout << "HiddenVertices: " << m_nb << " -> " << nb << std::endl;
        return (nb == m_nb);
    }
};

//-------------//
// CWCPosition //
//-------------//

template <class Scene, class Position, class Velocity>
class CLSPositions : public CLineSearch<Position, Velocity>
{
protected:
    Scene* m_scene;
    unsigned m_nb;

public:
    CLSPositions(Scene* scene,
                const unsigned max_iters,
                const double max_alpha)
    : CLineSearch<Position, Velocity>(max_iters, max_alpha)
    {
        m_scene = scene;
        m_nb = m_scene->count_visible_sites();
    }
    
    double compute_function() const
    {
        return ( - m_scene->compute_wcvt_energy() );
    }
    
    void compute_gradient(std::vector<Velocity>& V) const
    {
        m_scene->compute_position_gradient(V, -1.0);
    }
    
    bool update_scene(const std::vector<Position>& X)
    {
        if (m_scene->connectivity_fixed())
        {
            m_scene->clean_pixels();
            m_scene->assign_pixels();
            return true;
        }
        
        // `false` for the same reason as CLSWeights::update_scene above.
        m_scene->update_positions(X, true, false);
        return m_scene->update_triangulation(true);
        //return has_same_vertices();
    }
    
    bool has_same_vertices() const
    {
        unsigned nb = m_scene->count_visible_sites();
        if (nb != m_nb) std::cout << "HiddenVertices: " << m_nb << " -> " << nb << std::endl;
        return (nb == m_nb);
    }
};

#endif
