
(function() {
    'use strict';

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    
    const themeToggle = document.getElementById('theme-toggle');
    const navbar = document.getElementById('navbar');
    const backToTop = document.getElementById('back-to-top');
    const navLinks = document.querySelectorAll('.nav-link');
    
    const STORAGE_KEY = 'moonai-theme';
    
    // Scroll velocity tracking
    let lastScrollY = 0;
    let scrollVelocity = 0;
    let velocityTimeout = null;
    const FAST_SCROLL_THRESHOLD = 800; // pixels per second
    
    function getInitialTheme() {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            return stored;
        }
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
    
    function updateNavbar() {
        const scrollY = window.scrollY;
        if (scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }
    
    function updateBackToTop() {
        const scrollY = window.scrollY;
        if (scrollY > 500) {
            backToTop.classList.add('visible');
        } else {
            backToTop.classList.remove('visible');
        }
    }
    
    function scrollToTop() {
        if (prefersReducedMotion) {
            window.scrollTo(0, 0);
        } else {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
    }
    
    function handleNavLinkClick(e) {
        const href = this.getAttribute('href');
        if (href.startsWith('#')) {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const navHeight = navbar.offsetHeight;
                const targetPosition = target.getBoundingClientRect().top + window.scrollY - navHeight - 20;
                
                if (prefersReducedMotion) {
                    window.scrollTo(0, targetPosition);
                } else {
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            }
        }
    }
    
    function updateActiveNavLink() {
        const scrollPosition = window.scrollY + navbar.offsetHeight + 100;
        
        const sections = ['overview', 'details', 'technologies', 'documents'];
        
        for (const sectionId of sections) {
            const section = document.getElementById(sectionId);
            if (section) {
                const sectionTop = section.offsetTop;
                const sectionBottom = sectionTop + section.offsetHeight;
                
                if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                    navLinks.forEach(link => {
                        link.classList.remove('active');
                        if (link.getAttribute('href') === `#${sectionId}`) {
                            link.classList.add('active');
                        }
                    });
                    break;
                }
            }
        }
    }
    
    function updateScrollVelocity() {
        const currentScrollY = window.scrollY;
        const delta = Math.abs(currentScrollY - lastScrollY);
        scrollVelocity = delta;
        lastScrollY = currentScrollY;
        
        // Reset velocity after a short delay
        clearTimeout(velocityTimeout);
        velocityTimeout = setTimeout(() => {
            scrollVelocity = 0;
        }, 100);
    }
    
    function isFastScrolling() {
        return scrollVelocity > FAST_SCROLL_THRESHOLD;
    }
    
    function observeCards() {
        if (prefersReducedMotion || !('IntersectionObserver' in window)) {
            return;
        }
        
        const cards = document.querySelectorAll('.card');
        
        const observer = new IntersectionObserver((entries) => {
            // Skip animations if user is scrolling fast
            if (isFastScrolling()) {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        // Show immediately without animation
                        entry.target.classList.add('card-visible');
                        observer.unobserve(entry.target);
                    }
                });
                return;
            }
            
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Use CSS class instead of inline styles for better performance
                    entry.target.classList.add('card-visible');
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.3, // Increased from 0.1 to 0.3 for less frequent triggers
            rootMargin: '0px 0px -20px 0px' // Reduced margin
        });
        
        cards.forEach((card) => {
            // Start with animation class, no staggered delays
            card.classList.add('card-animate');
            observer.observe(card);
        });
    }
    
    function init() {
        const initialTheme = getInitialTheme();
        setTheme(initialTheme);
        
        if (themeToggle) {
            themeToggle.addEventListener('click', toggleTheme);
        }
        
        if (backToTop) {
            backToTop.addEventListener('click', scrollToTop);
        }
        
        navLinks.forEach(link => {
            link.addEventListener('click', handleNavLinkClick);
        });
        
        let ticking = false;
        
        function onScroll() {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    updateNavbar();
                    updateBackToTop();
                    updateActiveNavLink();
                    ticking = false;
                });
                ticking = true;
            }
            
            // Track scroll velocity separately (not in rAF)
            updateScrollVelocity();
        }
        
        window.addEventListener('scroll', onScroll, { passive: true });
        
        updateNavbar();
        updateBackToTop();
        updateActiveNavLink();
        
        // Initialize card observer after a short delay to not block initial render
        requestIdleCallback ? 
            requestIdleCallback(observeCards, { timeout: 500 }) :
            setTimeout(observeCards, 100);
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
