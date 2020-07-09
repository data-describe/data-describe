;(function($){
    "use strict";

    $(document).ready(function(){

        /*-------------------------------------------------
            Navbar fix
        --------------------------------------------------*/
        $(document).on('click','.navbar-area .navbar-nav li.menu-item-has-children>a',function(e){
            e.preventDefault();
        })  

        /*-------------------------------------------------
            wow js init
        --------------------------------------------------*/
        new WOW().init();

        /*-------------------------------------------------
            select onput
        --------------------------------------------------*/
        if ($('.select').length){
            $('.select').niceSelect();
        }

        /*--------------------------------------------------------
            Jarallax Active Code
        --------------------------------------------------------*/
        if ($.fn.jarallax) {
            $('.jarallax').jarallax({
                speed: 0.5
            });
        }

        /*-------------------------------------------------
            popover
        --------------------------------------------------*/
        if ($('[data-toggle="popover"]').length){
            $('[data-toggle="popover"]').popover();
        }

        /*------------------------------------
            Product Details Slider
        ------------------------------------*/
        var productDetailSlider = $('.single-thumbnail-slider');
        var pThumbanilSlider = $('.product-thumbnail-carousel');

        if (productDetailSlider.length) {
            productDetailSlider.slick({
                slidesToShow: 1,
                slidesToScroll: 1,
                arrows: false,
                fade: true,
                asNavFor: '.product-thumbnail-carousel'
              });
        }
        if (pThumbanilSlider.length) {
            pThumbanilSlider.slick({
                slidesToShow: 3,
                slidesToScroll: 1,
                asNavFor: '.single-thumbnail-slider',
                dots: false,
                centerMode: false,
                focusOnSelect: true,
                vertical: true,
                arrows:false,
                prevArrow: '<div class="slick-prev"><i class="fa fa-angle-double-up"></i></div>',
                nextArrow: '<div class="slick-next"><i class="fa fa-angle-double-down"></i></div>',
              });
        }


        /* -----------------------------------------------------
            main slider
        ----------------------------------------------------- */
        if ($('.banner-slider').length){
            $('.banner-slider').owlCarousel({
                items: 1,
                animateOut: 'fadeOut',
                animateIn: 'fadeIn',
                smartSpeed:450,
                loop: true,
                autoplay: true,
                autoplayTimeout: 10000,
                nav: true,
                dots: true,
                smartSpeed: 1500,
                navText:['<i class="ti-angle-left" aria-hidden="true"></i>','<i class="ti-angle-right" aria-hidden="true"></i>'],
            });
        }

        /* -----------------------------------------------------
            blog gallery slider
        ----------------------------------------------------- */
        if ($('.blog-gallery-slider').length){
            $('.blog-gallery-slider').owlCarousel({
                items: 1,
                smartSpeed:450,
                loop: true,
                autoplay: true,
                autoplayTimeout: 10000,
                nav: true,
                dots: false,
                smartSpeed: 1500,
                navText:['<i class="ti-angle-left" aria-hidden="true"></i>','<i class="ti-angle-right" aria-hidden="true"></i>'],
            });
        }

        /*-------------------------
            magnific popup activation
        -------------------------*/
        $('.video-play-btn,.video-popup,.small-vide-play-btn').magnificPopup({
            type: 'video',
            removalDelay: 260,
            mainClass: 'mfp-zoom-in',
        });

        /*------------------
            back to top
        ------------------*/
        $(document).on('click', '.back-to-top', function () {
            $("html,body").animate({
                scrollTop: 0
            }, 2000);
        });
        /*------------------------------
            counter section activation
        -------------------------------*/
        if($('.count-num').length){
            $('.count-num').counterUp({
                delay: 10,
                time: 5000
            });
        }

        /*-------------------------------
            Portfolio filter 
        ---------------------------------*/
        var $Container = $('.gallery-masonry');
        if ($Container.length > 0) {
            $('.gallery-masonry').imagesLoaded(function () {
                var festivarMasonry = $Container.isotope({
                    itemSelector: '.masonry-item', // use a separate class for itemSelector, other than .col-
                    masonry: {
                        gutter: 0
                    }
                });
                $(document).on('click', '.gallery-menu li', function () {
                    var filterValue = $(this).attr('data-filter');
                    festivarMasonry.isotope({
                        filter: filterValue
                    });
                });
            });
            $(document).on('click','.gallery-menu li' , function () {
                $(this).siblings().removeClass('active');
                $(this).addClass('active');
            });
        }
        
        /*-------------------------------------
            client carousel
        -------------------------------------*/
        var $clientCarousel = $('.client-slider');
        if ($clientCarousel.length > 0) {
            $clientCarousel.owlCarousel({
                loop: true,
                autoplay: false, //true if you want enable autoplay
                autoPlayTimeout: 1000,
                dots: true,
                nav: false,
                smartSpeed: 1500,
                responsive: {
                    0: {
                        items: 1
                    },
                    767: {
                        items: 1
                    },
                    768: {
                        items: 2
                    },
                    1200: {
                        items: 3,
                        center: true,
                    }
                }
            });
        }

        /*-------------------------------------
            client carousel 
        -------------------------------------*/
        var $clientCarousel2 = $('.client-slider-2');
        if ($clientCarousel2.length > 0) {
            $clientCarousel2.owlCarousel({
                loop: false,
                autoplay: false, //true if you want enable autoplay
                autoPlayTimeout: 1000,
                dots: false,
                nav: false,
                smartSpeed: 1500,
                responsive: {
                    0: {
                        items: 2
                    },
                    767: {
                        items: 2
                    },
                    768: {
                        items: 4
                    },
                    1200: {
                        items: 6
                    }
                }
            });
        }

        /*--------------------------------------
            testimonial-slider
        ---------------------------------------*/
        $("#testimonial-slider").AnimatedSlider( { 
            prevButton: "#btn_prev", 
            nextButton: "#btn_next",
            visibleItems: 5,
            infiniteScroll: true,
            willChangeCallback: function(obj, item) { $("#statusText").text("Will change to " + item); },
            changedCallback: function(obj, item) { $("#statusText").text("Changed to " + item); }
        });

        /*-------------------------------------
            Testimonial slider 2
        -------------------------------------*/
        var $testimonialslidertwo = $('.testimonial-slider-2');
        if ($testimonialslidertwo.length > 0) {
            $testimonialslidertwo.owlCarousel({
                loop: false,
                autoplay: false, //true if you want enable autoplay
                autoPlayTimeout: 1000,
                dots: true,
                nav: false,
                items: 1,
                animateOut: 'fadeOut',
                animateIn: 'fadeIn',
                smartSpeed: 1500,
            });
        }


        /*-------------------------------------
            screenshot slider
        -------------------------------------*/
        var $screenshotCarousel = $('.screenshot-slider');
        if ($screenshotCarousel.length > 0) {
            $screenshotCarousel.owlCarousel({
                loop: true,
                autoplay: false, //true if you want enable autoplay
                autoPlayTimeout: 1000,
                dots: false,
                nav: true,
                smartSpeed: 1000,
                navText:['<img src="assets/img/startup/left.svg" alt="arrow">','<img src="assets/img/startup/right.svg" alt="arrow">'],
                responsive: {
                    0: {
                        items: 1
                    },
                    767: {
                        items: 1
                    },
                    768: {
                        items: 3
                    },
                    1200: {
                        items: 5
                    }
                }
            });
        }


        /*-------------------------------------
            team slider
        -------------------------------------*/
        var $teamCarousel = $('.team-slider');
        if ($teamCarousel.length > 0) {
            $teamCarousel.owlCarousel({
                loop: true,
                autoplay: true, //true if you want enable autoplay
                autoPlayTimeout: 1000,
                dots: true,
                nav: true,
                items: 1,
                smartSpeed: 1500,
                animateOut: 'fadeOut',
                animateIn: 'fadeIn',
                navText:['<img src="assets/img/startup/left.svg" alt="arrow">','<img src="assets/img/startup/right.svg" alt="arrow">'],
            });
        }

        /*-------------------------------------
            marketing slider
        -------------------------------------*/
        var $screenshotCarousel = $('.marketing-slider');
        if ($screenshotCarousel.length > 0) {
            $screenshotCarousel.owlCarousel({
                loop: true,
                autoplay: true, //true if you want enable autoplay
                autoPlayTimeout: 1000,
                dots: false,
                margin: 16,
                nav: true,
                smartSpeed: 1500,
                navText:['<img src="assets/img/startup/left.svg" alt="arrow">','<img src="assets/img/startup/right.svg" alt="arrow">'],
                responsive: {
                    0: {
                        items: 1,
                        stagePadding: 60,
                    },
                    576: {
                        items: 2,
                        stagePadding: 60,
                    },
                    768: {
                        items: 2,
                        stagePadding: 50,
                    },
                    1300: {
                        items: 2,
                        stagePadding: 125,
                    }
                }
            });
        }

        /*--------------------------------------
            world map
        ---------------------------------------*/
        var jMap = $(".world-map"),
            height = jMap.height(),
            width = jMap.width(),
            mapJsonUrl = 'https://ucarecdn.com/8e1027ea-dafd-4d6c-bf1e-698d305d4760/world110m2.json',
            svg = d3.select(".world-map").append("svg")
            .attr("width", width)
            .attr("height", height);

        var getProjection = function(worldJson) {
            // create a first guess for the projection
            var scale = 1,
            offset = [ width / 2, height / 2 ],
            projection = d3.geoEquirectangular().scale( scale ).rotate( [0,0] ).center([0,5]).translate( offset ),
            bounds = mercatorBounds( projection ),
            scaleExtent;
          
            scale = width / (bounds[ 1 ][ 0 ] - bounds[ 0 ][ 0 ]);
            scaleExtent = [ scale, 10 * scale ];

            projection
            .scale( scaleExtent[ 0 ] );
          
          return projection;
        },
        mercatorBounds = function(projection) {
            // find the top left and bottom right of current projection
            var maxlat = 83,
            yaw = projection.rotate()[ 0 ],
            xymax = projection( [ -yaw + 180 - 1e-6, -maxlat ] ),
            xymin = projection( [ -yaw - 180 + 1e-6, maxlat ] );

           return [ xymin, xymax ];
        };
        d3.json(mapJsonUrl, function (error, worldJson) {
            if (error) throw error;
          
            var projection = getProjection(),
            path = d3.geoPath().projection( projection );
          
            svg.selectAll( 'path.land' )
            .data( topojson.feature( worldJson, worldJson.objects.countries ).features )
            .enter().append( 'path' )
            .attr( 'class', 'land' )
            .attr( 'd', path );
        });

        /*--------------------------------------
           form input Focus
        ---------------------------------------*/
        if($('.single-input').length){
            $('.single-input').on('focusin', function() {
                $(this).parent().find('label').addClass('active');
            });
            $('.single-input').on('focusout', function() {
                if (!this.value) {
                    $(this).parent().find('label').removeClass('active');
                }
            });
        }
        

        /*--------------------------------------
            pricing Active
        ---------------------------------------*/
        $(document).on('mouseover','.single-pricing',function() {
            $(this).addClass('single-pricing-active');
            $('.single-pricing').removeClass('single-pricing-active');
            $(this).addClass('single-pricing-active');
        });

        /*--------------------------------------
            cart close
        ---------------------------------------*/
        $('.cart-close span').on('click', function(){
            $(this).parents('tr').fadeOut();
        });
        $('.remove-product').on('click', function (e) {
            e.preventDefault();
            $(this).parent().fadeOut();
        });
        
       
        /*--------------------------------------------
            Search Popup
        ---------------------------------------------*/
        var bodyOvrelay =  $('#body-overlay');
        var searchPopup = $('#search-popup');

        $(document).on('click','#body-overlay',function(e){
            e.preventDefault();
        bodyOvrelay.removeClass('active');
            searchPopup.removeClass('active');
        });
        $(document).on('click','.search',function(e){
            e.preventDefault();
            searchPopup.addClass('active');
        bodyOvrelay.addClass('active');
        });

        /*--------------------------------------------
            Cart Popup
        ---------------------------------------------*/
        var bodyOvrelay =  $('#body-overlay');
        var cartPopup = $('#cart-popup');

        $(document).on('click','#body-overlay',function(e){
            e.preventDefault();
        bodyOvrelay.removeClass('active');
            cartPopup.removeClass('active');
        });
        $(document).on('click','#cart-btn',function(e){
            e.preventDefault();
            cartPopup.addClass('active');
        bodyOvrelay.addClass('active');
        });

        /*--------------------------------------------
            Cart Popup
        ---------------------------------------------*/
        var bodyOvrelay =  $('#body-overlay');
        var cartPopup = $('#cart-popup');

        $(document).on('click','#body-overlay',function(e){
            e.preventDefault();
        bodyOvrelay.removeClass('active');
            cartPopup.removeClass('active');
        });
        $(document).on('click','#cart-btn',function(e){
            e.preventDefault();
            cartPopup.addClass('active');
            bodyOvrelay.addClass('active');
        });

        /*--------------------------------------------
            signIn Popup
        ---------------------------------------------*/
        var bodyOvrelay =  $('#body-overlay');
        var signInPopup = $('#signIn-popup');

        $(document).on('click','#body-overlay',function(e){
            e.preventDefault();
            bodyOvrelay.removeClass('active');
            signInPopup.removeClass('active');
        });
        $(document).on('click','#signIn-btn',function(e){
            e.preventDefault();
            signInPopup.addClass('active');
            bodyOvrelay.addClass('active');
        });

        /*--------------------------------------------
            signUp Popup
        ---------------------------------------------*/
        var bodyOvrelay =  $('#body-overlay');
        var singupPopup = $('#signUp-popup');

        $(document).on('click','#body-overlay',function(e){
            e.preventDefault();
            bodyOvrelay.removeClass('active');
            singupPopup.removeClass('active');
        });
        $(document).on('click','#signUp-btn',function(e){
            e.preventDefault();
            singupPopup.addClass('active');
            bodyOvrelay.addClass('active');
        });


        /**---------------------------------------
         *  Progress BAR
         * -------------------------------------*/
        jQuery(window).on('scroll', function() {
            var windowHeight = $(window).height();
            function kalProgress() {
               var progress = $('.progress-rate');
               var len = progress.length;
                for (var i = 0; i < len; i++) {
                   var progressId = '#' + progress[i].id;
                   var dataValue = $(progressId).attr('data-value');
                   $(progressId).css({'width':dataValue+'%'});
                }
            }
            var progressRateClass = $('#progress-running');
             if ((progressRateClass).length) {
                 var progressOffset = $("#progress-running").offset().top - windowHeight;
                 if ($(window).scrollTop() > progressOffset) {
                     kalProgress();
                }
            }
        });

        /**---------------------------------------
         *  type js
        * -------------------------------------*/
        if($('.typed').length){
            var typed = $(".typed");
            $(function() {
                typed.typed({
                    strings: [" For Business."],
                    typeSpeed: 130,
                    loop: true
                });
            });
        }

        /**---------------------------------------
         *  file upload
        * -------------------------------------*/
        if($('.riyaqas-file-input').length){
            $(".riyaqas-file-input").on("change", function() {
              var fileName = $(this).val().split("\\").pop();
              $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
            });
        }


        /**---------------------------------------
         *  QTY Inputs
        * -------------------------------------*/
        $(function() {
            $("div.quantity").append('<a class="inc qty-button">+</a><a class="dec qty-button">-</a>');
            $(".qty-button").on("click", function() {
                console.log('clicked');
                var $button = $(this);
                var oldValue = $button.parent().find("input").val();

                if ($button.text() == "+") {
                    var newVal = parseFloat(oldValue) + 1;
                } else {
                    // Don't allow decrementing below zero
                    if (oldValue > 0) {
                    var newVal = parseFloat(oldValue) - 1;
                    } else {
                        newVal = 0;
                    }
                }
                $button.parent().find("input").val(newVal);
            });
        });

        /**---------------------------------------
         *  QTY Inputs
        * -------------------------------------*/
        if($('.slider-product-sorting').length){
            $( function() {
                $( ".slider-product-sorting" ).slider({
                range: true,
                min: 0,
                max: 90,
                values: [ 0, 90 ],
                slide: function( event, ui ) {
                $( "#amount" ).val( "$" + ui.values[ 0 ] + " - $" + ui.values[ 1 ] );
                }
                });
                $( "#amount" ).val( "$" + $( ".slider-product-sorting" ).slider( "values", 0 ) +
                " - $" + $( ".slider-product-sorting" ).slider( "values", 1 ) );
            } );
        }

    

    });


    //define variable for store last scrolltop
    var lastScrollTop = '';
    $(window).on('scroll', function () {
        //back to top show/hide
        var ScrollTop = $('.back-to-top');
        if ($(window).scrollTop() > 1000) {
           ScrollTop.fadeIn(1000);
        } else {
           ScrollTop.fadeOut(1000);
        }

        /*--------------------------
        sticky menu activation
       -------------------------*/
        if ($(window).scrollTop() >= 1) {
            $('.navbar-area').addClass('navbar-area-fixed');
        }
        else {
            $('.navbar-area').removeClass('navbar-area-fixed');
        }
       
    });
           

    $(window).on('load',function(){

        /*-----------------
            preloader
        ------------------*/
        var preLoder = $("#preloader");
        preLoder.fadeOut(1000);

        /*-----------------
            back to top
        ------------------*/
        var backtoTop = $('.back-to-top')
        backtoTop.fadeOut();

        /*---------------------
            Cancel Preloader
        ----------------------*/
        $(document).on('click','.cancel-preloader a',function(e){
            e.preventDefault();
            $("#preloader").fadeOut(2000);
        });

    });


    /* -------------------------------------------------------------
            Image Gallery Popup
    ------------------------------------------------------------- */
    function riyaqas_image_popup(selector){
        if ($(selector).length){
            $(selector).magnificPopup({
                delegate: 'a',
                type: 'image',
                gallery: { enabled: true },
                removalDelay: 500,
                callbacks: {
                    beforeOpen: function() {
                        this.st.image.markup = this.st.image.markup.replace('mfp-figure', 'mfp-figure mfp-with-anim');
                        this.st.mainClass = this.st.el.attr('data-effect');
                    }
                },
                closeOnContentClick: true,
                midClick: true
            });
        }
    }
    /* ------- Gallery image Popup--------- */
    riyaqas_image_popup('.gallery-masonry .masonry-item');


    /* -------------------------------------------------------------
        Audio Player
    ------------------------------------------------------------- */
    var
        sourcesSelector = document.body.querySelectorAll('select'),
        sourcesTotal = sourcesSelector.length
    ;

    for (var i = 0; i < sourcesTotal; i++) {
        sourcesSelector[i].addEventListener('change', function (e) {
            var
                media = e.target.closest('.media-container').querySelector('.mejs__container').id,
                player = mejs.players[media]
            ;

            player.setSrc(e.target.value.replace('&amp;', '&'));
            player.setPoster('');
            player.load();

        });

        // These media types cannot play at all on iOS, so disabling them
        if (mejs.Features.isiOS) {
            if (sourcesSelector[i].querySelector('option[value^="rtmp"]')) {
                sourcesSelector[i].querySelector('option[value^="rtmp"]').disabled = true;
            }
            if (sourcesSelector[i].querySelector('option[value$="webm"]')) {
                sourcesSelector[i].querySelector('option[value$="webm"]').disabled = true;
            }
            if (sourcesSelector[i].querySelector('option[value$=".mpd"]')) {
                sourcesSelector[i].querySelector('option[value$=".mpd"]').disabled = true;
            }
            if (sourcesSelector[i].querySelector('option[value$=".ogg"]')) {
                sourcesSelector[i].querySelector('option[value$=".ogg"]').disabled = true;
            }
            if (sourcesSelector[i].querySelector('option[value$=".flv"]')) {
                sourcesSelector[i].querySelector('option[value*=".flv"]').disabled = true;
            }
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        var mediaElements = document.querySelectorAll('video, audio'), total = mediaElements.length;

        for (var i = 0; i < total; i++) {
            new MediaElementPlayer(mediaElements[i], {
                pluginPath: 'https://cdn.jsdelivr.net/npm/mediaelement@4.2.7/build/',
                shimScriptAccess: 'always',
                success: function () {
                    var target = document.body.querySelectorAll('.player'), targetTotal = target.length;
                    for (var j = 0; j < targetTotal; j++) {
                        target[j].style.visibility = 'visible';
                    }
                }
            });
        }
    });


})(jQuery);
