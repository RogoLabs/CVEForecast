/**
 * Shared page shell: theme switching and the mobile nav.
 *
 * Loaded by every page. Pages that draw charts listen for the `themechange`
 * event rather than reimplementing any of this — Chart.js reads colours once,
 * when a chart is built, so it has to be told when the tokens move under it.
 */

(function shell() {
    const root = document.documentElement;

    /* Announce a theme change once, from one place, so a page with a canvas on
       it can repaint without every page owning its own detection. */
    const announce = () => document.dispatchEvent(new CustomEvent('themechange'));

    const toggle = document.getElementById('themeToggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            root.setAttribute('data-theme', next);
            try { localStorage.setItem('theme', next); } catch (e) { /* storage blocked */ }
        });
    }

    /* Covers the toggle above and anything else that stamps the attribute. */
    new MutationObserver(announce)
        .observe(root, { attributes: true, attributeFilter: ['data-theme'] });

    /* A visitor who has never chosen follows their OS. No attribute is stamped
       in that case, so the CSS has already followed along on its own and only
       a canvas needs telling. */
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        let pinned = false;
        try { pinned = !!localStorage.getItem('theme'); } catch (e) { /* storage blocked */ }
        if (!pinned) announce();
    });

    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            const open = navLinks.dataset.open === 'true';
            navLinks.dataset.open = String(!open);
            navToggle.setAttribute('aria-expanded', String(!open));
        });
    }
})();
