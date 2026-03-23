(function() {
    'use strict';

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

    const themeToggle = document.getElementById('theme-toggle');
    const backToTop = document.getElementById('back-to-top');
    const navLinks = document.querySelectorAll('.nav-link');
    const navbar = document.getElementById('navbar');

    const STORAGE_KEY = 'moonai-theme';

    // Scroll velocity tracking for card animations
    let lastScrollY = 0;
    let scrollVelocity = 0;
    let velocityTimeout = null;
    const FAST_SCROLL_THRESHOLD = 800;

    function getInitialTheme() {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) return stored;
        return prefersDarkScheme.matches ? 'dark' : 'light';
    }

    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(STORAGE_KEY, theme);
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        setTheme(next);
    }

    function updateBackToTop() {
        const scrollY = window.scrollY;
        backToTop?.classList.toggle('visible', scrollY > 500);
    }

    function scrollToTop() {
        if (prefersReducedMotion) {
            window.scrollTo(0, 0);
        } else {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

    function handleNavLinkClick(e) {
        const href = this.getAttribute('href');
        if (!href.startsWith('#')) return;

        e.preventDefault();
        const target = document.querySelector(href);
        if (!target) return;

        const navHeight = navbar.offsetHeight;
        const targetPosition = target.getBoundingClientRect().top + window.scrollY - navHeight - 20;

        if (prefersReducedMotion) {
            window.scrollTo(0, targetPosition);
        } else {
            window.scrollTo({ top: targetPosition, behavior: 'smooth' });
        }
    }

    function updateActiveNavLink() {
        const scrollPosition = window.scrollY + navbar.offsetHeight + 100;
        const sections = ['overview', 'details', 'technologies', 'documents'];

        for (const sectionId of sections) {
            const section = document.getElementById(sectionId);
            if (!section) continue;

            const sectionTop = section.offsetTop;
            const sectionBottom = sectionTop + section.offsetHeight;

            if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === `#${sectionId}`);
                });
                break;
            }
        }
    }

    function updateScrollVelocity() {
        const currentScrollY = window.scrollY;
        const delta = Math.abs(currentScrollY - lastScrollY);
        scrollVelocity = delta;
        lastScrollY = currentScrollY;

        clearTimeout(velocityTimeout);
        velocityTimeout = setTimeout(() => { scrollVelocity = 0; }, 100);
    }

    function isFastScrolling() {
        return scrollVelocity > FAST_SCROLL_THRESHOLD;
    }

    function observeCards() {
        if (prefersReducedMotion || !('IntersectionObserver' in window)) return;

        const cards = document.querySelectorAll('.card');
        const observer = new IntersectionObserver((entries) => {
            if (isFastScrolling()) {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('card-visible');
                        observer.unobserve(entry.target);
                    }
                });
                return;
            }

            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('card-visible');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.3,
            rootMargin: '0px 0px -20px 0px'
        });

        cards.forEach(card => {
            card.classList.add('card-animate');
            observer.observe(card);
        });
    }

    function init() {
        setTheme(getInitialTheme());

        themeToggle?.addEventListener('click', toggleTheme);
        backToTop?.addEventListener('click', scrollToTop);
        navLinks.forEach(link => link.addEventListener('click', handleNavLinkClick));

        let ticking = false;
        window.addEventListener('scroll', () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    updateBackToTop();
                    updateActiveNavLink();
                    ticking = false;
                });
                ticking = true;
            }
            updateScrollVelocity();
        }, { passive: true });

        updateBackToTop();
        updateActiveNavLink();

        (window.requestIdleCallback || (cb => setTimeout(cb, 100)))(observeCards);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
