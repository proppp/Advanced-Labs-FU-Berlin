%%%%%%%%%%%%%%%%%%
% Creators: Wout Dillen & Marijn Koolen
% 
% Version History:
% - v1.0: 2019/09/04
% - v1.1: 2021/07/01
% - v2.0: 2022/05/06
% - v2.1: 2024/10/31 -- Happy Halloween!
% - v2.1.0: 2024/10/31
% - v?
%
% Revision history:
%
% - v1.0 -> v1.1
%       ltfamily is no longer defined, switch renewcommand to newcommand
% - v1.1 -> v2.0
%       automatically archives new releases in Zenodo
%       changelog: https://github.com/DHBenelux/jDHBenelux-author-template/releases/tag/v2.0
% - v2.0 -> v2.1
%       hyperref package moved up to fix hyperxmp issue
%       added lmodern to fix palatino issue with missing font shapes
%       removed fancyhdr which was no longer used, and caused issues with KOMAfont
% - v2.1 -> v2.1.0
%       no changes in the repository, only fixed Zenodo synchronisation
% - v2.1.0 -> v?
%       fixed small typo in revision history, and added v2.1.0 update
%       ...
%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dhbenelux}[2021/07/01 DH Benelux Journal template]

\LoadClass[onecolumn]{scrartcl}


 
%%%%%%%%%%%%%%%%%%
% classoptions
%%%%%%%%%%%%%%%%%%
\KOMAoptions{
    titlepage=false,
    paper=210mm:297mm,          % A4 dimensions
    twoside=false, 
    twocolumn=false,
    parskip=false,              % space between paragraphs. "full" gives more space; "false" uses indentation instead
    %headings=small,
    draft=false,                % switch to "true" to activate draftmode
}

%%%%%%%%%%%%%%%%%%
% pagesize
%%%%%%%%%%%%%%%%%%
\usepackage[
    top=23mm,
    left=34mm,
    height=237mm,
    width=143mm,
    ]{geometry}

%

%%%%%%%%%%%%%%%%%%
% packages
%%%%%%%%%%%%%%%%%%
\usepackage[x11names,table,dvipsnames]{xcolor}
\usepackage[hyphens]{url}                       % simple URL typesetting
\usepackage{footmisc}                           % Make sure this is loaded before hyperref to avoid breaking footnote hyperlinks    
\usepackage[unicode=true,pdfusetitle,           % fixes hyperref issue
   bookmarks=true, 
   bookmarksnumbered=false, 
   bookmarksopen=false,
   breaklinks=false, 
   pdfborder={0 0 1}, 
   backref=false, 
   colorlinks=false]{hyperref}
\usepackage{booktabs}                           % professional-quality tables
\usepackage{nicefrac}                           % compact symbols for 1/2, etc.
\usepackage{microtype}                          % microtypography
\usepackage{lipsum}                             % lorem ipsum at the ready
\usepackage{graphicx}                           % for figures
\usepackage{footmisc}                           % makes symbol footnotes possible
\usepackage{ragged2e}
%\usepackage{changepage}                        % detect odd/even pages
\usepackage{array}
\usepackage{float}                              % get figures etc. to stay where they are with [H]
%\usepackage{subfigure}                         % \subfigures witin a \begin{figure}
\usepackage[textsize=footnotesize]{todonotes}   % enables \todo's for editors
\usepackage{etoolbox}                           % supplies commands like \AtBeginEnvironment and \atEndEnvironment
\usepackage{ifdraft}                            % switches on proofreading options in the draft mode
\usepackage{tikz}                               % for copyright statement
\usepackage[absolute]{textpos}                  % for copyright statement
\usepackage[                                    % for copyright statement
    type={CC},
    modifier={by},
    version={4.0},
]{doclicense}

%%%%%%%%%%%%%%%%%%
% fix Hyperref (cont.)
%%%%%%%%%%%%%%%%%%
\hypersetup{pdfauthor={Wout Dillen}}            
% requires hyperref package to be defined (see 'Packages' above)
% fix originally found at: https://albertsk.org/2019/10/10/how-to-avoid-warning-messages-from-hyperref-package-in-latex/
% FYI: the pdfauthor name is irrelevant, it just needs to have one. Feel free to change to another name.
% moved the hyperref package up to fix further hyperxmp issue (v2.1 fix)
% // WD


%%%%%%%%%%%%%%%%%%
% encoding
%%%%%%%%%%%%%%%%%%
\usepackage[L7x, T1]{fontenc}           % use 8-bit T1 fonts
\usepackage[utf8]{inputenc}             % allow utf-8 input
% languages
\usepackage[main=english, lithuanian, latin]{babel}
% special characters
\DeclareUnicodeCharacter{2D7}{}         % allow ¡ character
\DeclareUnicodeCharacter{45E}{}         % allow ў character
\DeclareUnicodeCharacter{2032}{}        % allow ′ character
\DeclareUnicodeCharacter{22EE}{}        % allow ⋮ character
\usepackage{textalpha}                  % allows for greek characters in text e.g. \textalpha = α
\usepackage{textcomp}                   % allows \textrightarrow etc.
\usepackage{authblk}                    % for multiple authors and affiliations


%%%%%%%%%%%%%%%%%%
% Fonts
%%%%%%%%%%%%%%%%%%
\usepackage{lmodern}                    % fixes issue when font shapes can't be found in palatino
% Palatino font options
\usepackage{mathpazo}                   % Palatino in LaTeX math
\usepackage{tgpagella}                  % Palatino font
\newcommand{\ltfamily}{\familydefault}  % Makes sure LaTeX keeps using Palatino in Lithuanian 
\addtokomafont{disposition}{\rmfamily}  % Palatino for titles etc.
\setkomafont{descriptionlabel}{         % font for description lists    
    \usekomafont{captionlabel}\bfseries % Palatino bold
}
\setkomafont{caption}{\footnotesize}    % smaller font size for captions

%%%%%%%%%%%%%%%%%%
% Title page
%%%%%%%%%%%%%%%%%%
\date{}                                     % Get rid of date ... there should be a neater way to do this

%set unit to be pagewidth and height, and increase inner margin of box
\setlength{\TPHorizModule}{\paperwidth}\setlength{\TPVertModule}{\paperheight}
\TPMargin{15pt} % Margin from text content

% Below is taken from https://tex.stackexchange.com/questions/55813/how-to-add-copyright-notice-in-a-box-with-borders-at-bottom-of-first-page
% Wout modified it using \usepackage{doclicense} (http://ctan.triasinformatica.nl/macros/latex/contrib/doclicense/doclicense.pdf)
% Define \copyrightstatement command for easier use
\newcommand{\copyrightstatement}{
    \begin{textblock}{0.84}(0.08,0.02)      % tweak here: {box width}(leftposition, rightposition)
         \noindent
         \footnotesize
         \centering
         \doclicenseIcon \the\year.         % alternative: \doclicenseImage[imagewidth=4em] – Wout thinks icons are neater.
         \doclicenseLongText                % includes a link
         
    \end{textblock}
}

\newcommand{\keywords}[1]{\textbf{Keywords:} #1}

%%%%%%%%%%%%%%%%%%
% Bibliography
%%%%%%%%%%%%%%%%%%
\usepackage[sort&compress,semicolon,authoryear]{natbib} % for citations
\bibliographystyle{plainnat}


%%%%%%%%%%%%%%%%%%
% Spacing options
%%%%%%%%%%%%%%%%%%

%\renewcommand{\bigskip}{\vpsace{4em}}
%\renewcommand{\medskip}{\vpsace{2em}}


%%%%%%%%%%%%%%%%%%
% Environments
%%%%%%%%%%%%%%%%%%
\sloppy                         % preferences more space between words over overrunning margins
\lefthyphenmin=3                % suppresses hyphenation after only 1 or 2 characters
% NB: You will need to repeat \lefthyphenmin in the text if you use \selectlanguage    
\AtBeginEnvironment{quote}{\footnotesize\vskip 1em}
\AtEndEnvironment{quote}{\vskip 1em}

\setkomafont{caption}{\footnotesize}

% Motto
\newenvironment{motto}{% 
\setlength{\parindent}{0pt} \small\raggedleft}{%
\vskip 2em
}{}


%%%%%%%%%%%%%%%%%%
% Sectionoptions
%%%%%%%%%%%%%%%%%%
% command for centering section headings
\newcommand{\centerheading}[1]{   
    \hspace*{\fill}#1\hspace*{\fill}
}

% Remove "Part #." from \part titles
% KOMA default: \newcommand*{\partformat}{\partname~\thepart\autodot}
\renewcommand*{\partformat}{} 

% No dots after figure or table numbers
\renewcommand*{\figureformat}{\figurename~\thefigure}
\renewcommand*{\tableformat}{\tablename~\thetable}

% paragraph handling
\setparsizes%
    {1em}% indent
    {0pt}% maximum space between paragraphs
    {0pt plus 1fil}% last line not justified
    

% In the "Authors" section, author names are put in the \paragraph{} headings. To reduce the space after these  headings, the default {-1em} has been changed to {-.4em} below.
\makeatletter
\renewcommand\paragraph{\@startsection {paragraph}{4}{\z@ }{3.25ex \@plus 1ex \@minus .2ex}{-.4em}{\normalfont \normalsize \bfseries }
}
\makeatother

% add the following (uncommented) in environments where you want to count paragraph numbers in the margin
%    \renewcommand*{\paragraphformat}{%
%    \makebox[-4pt][r]{\footnotesize\theparagraph\autodot\enskip}
%    }
%    \renewcommand{\theparagraph}{\arabic{paragraph}}
%    \setcounter{paragraph}{0}
%    \setcounter{secnumdepth}{4}
    
%%%%%%%%%%%%%%%%%%
% footnoteformat
%%%%%%%%%%%%%%%%%%
\renewcommand{\footnoterule}{%
    \kern .5em  % call this kerna
    \hrule height 0.4pt width .2\columnwidth    % the .2 value made the footnote ruler (horizontal line) smaller (was at .4)
    \kern .5em % call this kernb
}
\renewcommand{\footnotelayout}{
    \hspace{1.5em}                              % space between footnote mark and footnote text
}    
\newcommand{\mytodo}[1]{\textcolor{red}{#1}}

%%%%%%%%%%%%%%%%%%
% Proofs
%%%%%%%%%%%%%%%%%%
% Proofs watermark
%\usepackage{draftwatermark}
%\SetWatermarkText{PROOFS}
% Line numbers
%\usepackage[switch, modulo, pagewise]{lineno}
%\linenumbers

\makeatletter
 \newsavebox{\@linebox}
 \savebox{\@linebox}[3em][t]{\parbox[t]{3em}{%
   \@tempcnta\@ne\relax
   \loop{\underline{\scriptsize\the\@tempcnta}}\\
     \advance\@tempcnta by \@ne\ifnum\@tempcnta<48\repeat}}
\makeatother



%%%%%%%%%%%%%%%%%%
% code snippets
%%%%%%%%%%%%%%%%%%
% colours for code notations
\usepackage{listings}       
	\renewcommand\lstlistingname{Listing} 
	\lstset{                    % basic formatting (bash etc.)
	       basicstyle=\ttfamily,
 	       showstringspaces=false,
	       commentstyle=\color{BrickRed},
	       keywordstyle=\color{RoyalBlue}
	}
	\lstdefinelanguage{XML}{     % specific XML formatting overrides
		  basicstyle=\ttfamily,
		  morestring=[s]{"}{"},
		  morecomment=[s]{?}{?},
		  morecomment=[s]{!--}{--},
		  commentstyle=\color{OliveGreen},
		  moredelim=[s][\color{Black}]{>}{<},
		  moredelim=[s][\color{RawSienna}]{\ }{=},
		  stringstyle=\color{RoyalBlue},
 		  identifierstyle=\color{Plum}
	}
    % HOW TO USE? BASH EXAMPLE
    %   \begin{lstlisting}[language=bash]
    %   #some comment
    %   cd Documents
    %   \end{lstlisting}



